import os

import torch
import torchvision.transforms as T
from diffusers.pipeline_utils import DiffusionPipeline
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from src.utils.image_composition import compose_img, compose_img_dresscode


@torch.inference_mode()
def generate_images_from_mgd_pipe(
    test_order: bool,
    pipe: DiffusionPipeline,
    test_dataloader: DataLoader,
    save_name: str,
    dataset: str,
    output_dir: str,
    guidance_scale: float = 7.5,
    guidance_scale_pose: float = 7.5,
    guidance_scale_sketch: float = 7.5,
    sketch_cond_rate: float = 1.0,
    start_cond_rate: float = 0.0,
    no_pose: bool = False,
    disentagle: bool = False,
    seed: int = 1234,
    pose_abl: bool = False,
    sktech_abl: bool = False,
) -> None:
    # This function generates images from the given test dataloader and saves them to the output directory.
    """
    Args:
        test_order: The order of the test dataset.
        pipe: The diffusion pipeline.
        test_dataloader: The test dataloader.
        save_name: The name of the saved images.
        dataset: The name of the dataset.
        output_dir: The output directory.
        guidance_scale: The guidance scale.
        guidance_scale_pose: The guidance scale for the pose.
        guidance_scale_sketch: The guidance scale for the sketch.
        sketch_cond_rate: The sketch condition rate.
        start_cond_rate: The start condition rate.
        no_pose: Whether to use the pose.
        disentagle: Whether to use disentagle.
        seed: The seed.

        Returns:
        None
    """
    assert save_name != "", "save_name must be specified"
    assert output_dir != "", "output_dir must be specified"

    path = os.path.join(output_dir, f"{save_name}_{test_order}", "images")

    os.makedirs(path, exist_ok=True)
    generator = torch.Generator("cuda").manual_seed(seed)

    for batch in tqdm(test_dataloader):
        model_img = batch["image"]
        mask_img = batch["inpaint_mask"]
        mask_img = mask_img.type(torch.float32)
        prompts = batch[
            "original_captions"
        ]  # prompts is a list of length N, where N=batch size.
        pose_map = batch["pose_map"]
        sketch = batch["im_sketch"]
        ext = ".jpg"

        if pose_abl and not sketch_abl:
            pose_map[:] = pose_map[0]
        elif sketch_abl and not pose_abl:
            sketch[:] = sketch[0]
        elif pose_abl and sketch_abl:
            raise ValueError("pose_abl and sketch_abl cannot be True at the same time")

        if disentagle:
            guidance_scale = guidance_scale
            num_samples = 1
            guidance_scale_pose = guidance_scale_pose
            guidance_scale_sketch = guidance_scale_sketch
            generated_images = pipe(
                prompt=prompts,
                image=model_img,
                mask_image=mask_img,
                pose_map=pose_map,
                sketch=sketch,
                height=512,
                width=384,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_samples,
                generator=generator,
                sketch_cond_rate=sketch_cond_rate,
                guidance_scale_pose=guidance_scale_pose,
                guidance_scale_sketch=guidance_scale_sketch,
                start_cond_rate=start_cond_rate,
                no_pose=no_pose,
            ).images
        else:
            guidance_scale = 7.5
            num_samples = 1
            generated_images = pipe(
                prompt=prompts,
                image=model_img,
                mask_image=mask_img,
                pose_map=pose_map,
                sketch=sketch,
                height=512,
                width=384,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_samples,
                generator=generator,
                sketch_cond_rate=sketch_cond_rate,
                start_cond_rate=start_cond_rate,
                no_pose=no_pose,
            ).images

            for i in range(len(generated_images)):
                model_i = model_img[i] * 0.5 + 0.5
                if dataset == "vitonhd":
                    final_img = compose_img(
                        model_i, generated_images[i], batch["im_parse"][i]
                    )
                else:  # dataset == Dresscode
                    face = batch["stitch_label"][i].to(model_img.device)
                    face = T.functional.resize(
                        face,
                        size=(512, 384),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True,
                    )

                    final_img = compose_img_dresscode(
                        gt_img=model_i,
                        fake_img=T.functional.to_tensor(generated_images[i]).to(
                            model_img.device
                        ),
                        im_head=face,
                    )

                final_img = T.functional.to_pil_image(final_img)
                model_image = T.functional.to_pil_image(model_i)
                mask_image_i = T.functional.to_pil_image(mask_img[i])
                pose_map_i = pose_map[i].max(dim=0)[0]
                pose_map_i = T.functional.to_pil_image(pose_map_i)
                sketch_i = T.functional.to_pil_image(sketch[i])

                text_image = Image.new('RGB', (model_image.width, model_image.height), color = 'black')
                draw = ImageDraw.Draw(text_image)
                font = ImageFont.load_default()
                text = prompts[i].replace(",", "\n")

                lines = text.split('\n')
                max_width = max(draw.textsize(line, font=font)[0] for line in lines)
                total_height = sum(draw.textsize(line, font=font)[1] for line in lines)

                # Starting Y position
                width = model_image.width
                height = model_image.height
                y = (height - total_height) / 2

                # Draw each line of text
                for line in lines:
                    line_width, line_height = draw.textsize(line, font=font)
                    x = (width - line_width) / 2
                    draw.text((x, y), line, fill="white", font=font)
                    y += line_height

                text_width, text_height = draw.textsize(text, font=font)

                new_width = model_image.width * 3
                new_height = height * 2

                concat_image = Image.new("RGB", (new_width, new_height))

                concat_image.paste(text_image, (0, 0))
                concat_image.paste(mask_image_i, (width, 0))
                concat_image.paste(model_image, (width*2, 0))
                concat_image.paste(pose_map_i, (0, height))
                concat_image.paste(sketch_i, (width, height))
                concat_image.paste(final_img, (width*2, height))

                concat_image.save(
                    os.path.join(path, batch["im_name"][i].replace(".jpg", ext))
                )
