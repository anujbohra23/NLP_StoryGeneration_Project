Place the LoRA adapter file `adapter_model.safetensors` here before running the fine-tuned story generator.

It was intentionally not duplicated into this lightweight handoff zip. In the original uploaded project, it is located at:
`fine_tuning/Checkpoint_100/adapter_model.safetensors`

The pipeline can still be tested without it by using:
`python src/final_storybook_pipeline.py --input-story examples/sample_existing_story.json --out outputs/sample_story --skip-images`
