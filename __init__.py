import os
import re
import base64
import logging
import numpy as np
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
from twelvelabs import TwelveLabs
from twelvelabs.types.video_context import VideoContext_Base64String

SIMILARITY_THRESHOLD = 0.1  # below this -> MISMATCH

logger = logging.getLogger("fiftyone.plugins.twelvelabs_label_checker")
logger.setLevel(logging.INFO)


def log(msg):
    logger.info(msg)
    print(f"[TwelveLabs Plugin] {msg}", flush=True)


def register(p):
    p.register(CheckVideoLabel)


class CheckVideoLabel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="check_video_label",
            label="Check Video Label with Twelve Labs",
            description="Uses Marengo embeddings to verify if the ground truth label matches video content",
            dynamic=True,
            execute_as_generator=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "label_field",
            label="Label Field",
            description="Which field holds the ground truth label? (e.g. 'ground_truth')",
            default="ground_truth",
            required=True,
        )
        inputs.str(
            "info",
            label="Info",
            description=f"Will process {len(ctx.selected)} selected sample(s) using Twelve Labs Marengo",
        )
        return types.Property(inputs, view=types.View(label="Check Video Label"))

    def execute(self, ctx):
        tl_api_key = ctx.secrets.get("TWELVELABS_API_KEY") or os.environ.get("TWELVELABS_API_KEY")
        label_field = ctx.params.get("label_field", "ground_truth")

        log(f"Starting operator. API key present: {bool(tl_api_key)}")
        log(f"Label field: '{label_field}'")

        tl_client = TwelveLabs(api_key=tl_api_key)
        log("TwelveLabs client initialized")

        selected_ids = ctx.selected
        if not selected_ids:
            log("ERROR: No samples selected")
            yield ctx.set_progress(label="No samples selected!", progress=1.0)
            return

        total = len(selected_ids)
        log(f"Processing {total} sample(s)")
        results = []

        for i, sample_id in enumerate(selected_ids):
            sample = ctx.dataset[sample_id]
            video_path = sample.filepath
            log(f"--- Sample {i+1}/{total} (id={sample_id}) ---")
            log(f"  Video path: {video_path}")

            yield ctx.set_progress(
                label=f"Processing sample {i+1}/{total}...",
                progress=(i / total),
            )

            # Get existing ground truth label
            label_obj = sample.get_field(label_field)
            if label_obj is None:
                log(f"  ERROR: Field '{label_field}' is None — skipping")
                sample["label_check"] = "ERROR"
                sample["is_mislabeled"] = False
                sample.save()
                continue

            # Handle both fo.Classification and plain string labels
            if hasattr(label_obj, "label"):
                ground_truth_label = label_obj.label
            else:
                ground_truth_label = str(label_obj)

            # Clean label -- "PlayingGuitar" -> "playing guitar"
            label_text = re.sub(r'(?<!^)(?=[A-Z])', ' ', ground_truth_label).lower().strip()
            log(f"  Ground truth: '{ground_truth_label}' -> cleaned: '{label_text}'")

            try:
                # --- STEP 1: Embed the video using Marengo (async) ---
                yield ctx.set_progress(
                    label=f"Sample {i+1}/{total}: Uploading video...",
                    progress=(i / total),
                )
                log(f"  Uploading video to Twelve Labs...")
                with open(video_path, "rb") as vf:
                    video_task = tl_client.embed.tasks.create(
                        model_name="marengo3.0",
                        video_file=vf,
                    )
                log(f"  Video upload done. Task ID: {video_task.id}")

                yield ctx.set_progress(
                    label=f"Sample {i+1}/{total}: Waiting for video embedding...",
                    progress=(i + 0.2) / total,
                )
                log(f"  Waiting for video embedding to complete...")
                tl_client.embed.tasks.wait_for_done(video_task.id, sleep_interval=3)
                log(f"  Video embedding complete. Retrieving result...")
                video_result = tl_client.embed.tasks.retrieve(video_task.id)
                video_embedding = np.array(
                    video_result.video_embedding.segments[0].float_
                )
                log(f"  Video embedding retrieved ({len(video_embedding)}-dim vector)")

                # --- STEP 2: Embed the label text using same Marengo model (sync) ---
                yield ctx.set_progress(
                    label=f"Sample {i+1}/{total}: Embedding label text...",
                    progress=(i + 0.4) / total,
                )
                log(f"  Embedding label text: '{label_text}'...")
                text_result = tl_client.embed.create(
                    model_name="marengo3.0",
                    text=label_text,
                )
                text_embedding = np.array(
                    text_result.text_embedding.segments[0].float_
                )
                log(f"  Text embedding retrieved ({len(text_embedding)}-dim vector)")

                # --- STEP 3: Cosine similarity ---
                similarity = float(
                    np.dot(video_embedding, text_embedding)
                    / (np.linalg.norm(video_embedding) * np.linalg.norm(text_embedding))
                )

                verdict = "MATCH" if similarity >= SIMILARITY_THRESHOLD else "MISMATCH"
                is_mislabeled = verdict == "MISMATCH"
                log(f"  Similarity: {similarity:.4f} -> {verdict}")

                # --- STEP 4: Get video description using Twelve Labs analyze ---
                yield ctx.set_progress(
                    label=f"Sample {i+1}/{total}: Generating description...",
                    progress=(i + 0.7) / total,
                )
                log(f"  Generating video description...")
                with open(video_path, "rb") as vf:
                    video_b64 = base64.b64encode(vf.read()).decode("utf-8")
                desc_result = tl_client.analyze(
                    prompt="Describe what is happening in this video in one sentence.",
                    video=VideoContext_Base64String(
                        type="base64_string",
                        base_64_string=video_b64,
                    ),
                )
                video_description = desc_result.data
                log(f"  Video description: '{video_description}'")

            except Exception as e:
                log(f"  ERROR: {type(e).__name__}: {e}")
                sample["label_check"] = f"ERROR: {e}"
                sample["is_mislabeled"] = False
                sample["similarity_score"] = -1.0
                sample["video_description"] = fo.Classification(label="ERROR")
                sample.save()
                continue

            # --- Write results back to sample ---
            sample["similarity_score"] = similarity
            sample["label_check"] = verdict
            sample["is_mislabeled"] = is_mislabeled
            sample["video_description"] = fo.Classification(label=video_description)
            sample.save()
            log(f"  Saved to sample: score={similarity:.4f}, verdict={verdict}, description='{video_description}'")

            results.append(
                f'Ground truth: "{ground_truth_label}"\n'
                f'AI description: "{video_description}"\n'
                f"Score: {similarity:.2f} → {verdict}"
            )

        separator = "\n" + "─" * 40 + "\n"
        summary = f"Done! {len(results)} sample(s) processed.\n{separator}" + separator.join(results)
        log(summary)

        yield ctx.set_progress(label="Reloading dataset...", progress=0.95)
        ctx.trigger("reload_dataset")
        ctx.trigger("reload_samples")

        yield ctx.set_progress(label=summary, progress=1.0)
