import base64
import csv
import dataclasses
import io
import json
import math
import os
import re
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext

import cv2  # Add OpenCV import at top if not already imported
import matplotlib.cm as cm
import numpy as np
from google import genai
from google.genai import types
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

# Additional colors for masks
additional_colors = [
    colorname for (colorname, colorcode) in ImageColor.colormap.items()
]

# Gemini models with descriptions
MODELS_INFO = {
    "gemini-2.5-flash-preview-05-20": {
        "name": "Gemini 2.5 Flash Preview 05-20",
        "desc": "Audio, images, videos, and text. Adaptive thinking, cost efficiency.",
    },
    "gemini-2.5-flash-preview-native-audio-dialog": {
        "name": "Gemini 2.5 Flash Native Audio Dialog",
        "desc": "Audio, videos, and text. High quality natural conversational audio outputs.",
    },
    "gemini-2.5-flash-exp-native-audio-thinking-dialog": {
        "name": "Gemini 2.5 Flash Exp Native Audio Thinking Dialog",
        "desc": "Audio, videos, and text. High quality natural conversational audio outputs.",
    },
    "gemini-2.5-flash-preview-tts": {
        "name": "Gemini 2.5 Flash Preview TTS",
        "desc": "Text to audio. Low latency, controllable text-to-speech.",
    },
    "gemini-2.5-pro-preview-05-06": {
        "name": "Gemini 2.5 Pro Preview",
        "desc": "Audio, images, videos, and text. Enhanced thinking, reasoning, multimodal understanding.",
    },
    "gemini-2.5-pro-preview-tts": {
        "name": "Gemini 2.5 Pro Preview TTS",
        "desc": "Text to audio. Low latency, controllable text-to-speech.",
    },
    "gemini-2.0-flash": {
        "name": "Gemini 2.0 Flash",
        "desc": "Audio, images, videos, and text. Next gen features, speed, realtime streaming.",
    },
    "gemini-2.0-flash-preview-image-generation": {
        "name": "Gemini 2.0 Flash Preview Image Generation",
        "desc": "Audio, images, videos, and text. Conversational image generation and editing.",
    },
    "gemini-2.0-flash-lite": {
        "name": "Gemini 2.0 Flash Lite",
        "desc": "Audio, images, videos, and text. Cost efficiency and low latency.",
    },
    "gemini-1.5-flash": {
        "name": "Gemini 1.5 Flash",
        "desc": "Audio, images, videos, and text. Fast and versatile performance.",
    },
    "gemini-1.5-flash-8b": {
        "name": "Gemini 1.5 Flash 8B",
        "desc": "Audio, images, videos, and text. High volume and lower intelligence tasks.",
    },
    "gemini-1.5-pro": {
        "name": "Gemini 1.5 Pro",
        "desc": "Audio, images, videos, and text. Complex reasoning tasks.",
    },
    "gemini-embedding-exp": {
        "name": "Gemini Embedding",
        "desc": "Text embeddings. Measuring relatedness of text strings.",
    },
    "imagen-3.0-generate-002": {
        "name": "Imagen 3",
        "desc": "Text to images. Advanced image generation.",
    },
    "veo-2.0-generate-001": {
        "name": "Veo 2",
        "desc": "Text, images to video. High quality video generation.",
    },
    "gemini-2.0-flash-live-001": {
        "name": "Gemini 2.0 Flash Live",
        "desc": "Audio, video, text. Low-latency bidirectional voice and video interactions.",
    },
}


@dataclasses.dataclass(frozen=True)
class SegmentationMask:
    y0: int
    x0: int
    y1: int
    x1: int
    mask: np.ndarray
    label: str


def parse_json(json_output: str) -> str:
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i + 1 :])
            json_output = json_output.split("```")[0]
            break
    return json_output


def parse_segmentation_masks(
    predicted_str: str, img_height: int, img_width: int
) -> list[SegmentationMask]:
    items = json.loads(parse_json(predicted_str))
    masks = []
    for item in items:
        raw_box = item["box_2d"]
        abs_y0 = int(raw_box[0] / 1000 * img_height)
        abs_x0 = int(raw_box[1] / 1000 * img_width)
        abs_y1 = int(raw_box[2] / 1000 * img_height)
        abs_x1 = int(raw_box[3] / 1000 * img_width)

        if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
            print("Invalid bounding box", raw_box)
            continue

        label = item.get("label", "")

        png_str = item.get("mask", "")
        if not png_str.startswith("data:image/png;base64,"):
            print("Invalid mask encoding")
            continue
        png_str = png_str.removeprefix("data:image/png;base64,")
        png_bytes = base64.b64decode(png_str)

        mask_img = Image.open(io.BytesIO(png_bytes)).convert("L")
        bbox_height = abs_y1 - abs_y0
        bbox_width = abs_x1 - abs_x0
        if bbox_height < 1 or bbox_width < 1:
            print("Invalid bbox size")
            continue

        mask_img = mask_img.resize(
            (bbox_width, bbox_height), resample=Image.Resampling.BILINEAR
        )
        np_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        np_mask[abs_y0:abs_y1, abs_x0:abs_x1] = np.array(mask_img)
        masks.append(SegmentationMask(abs_y0, abs_x0, abs_y1, abs_x1, np_mask, label))
    return masks


def overlay_mask_on_img(
    img: Image.Image, mask: np.ndarray, color: str, alpha: float = 0.7
) -> Image.Image:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")

    try:
        color_rgb = ImageColor.getrgb(color)
    except ValueError as e:
        raise ValueError(f"Invalid color name '{color}'. Error: {e}")

    img_rgba = img.convert("RGBA")
    width, height = img_rgba.size

    alpha_int = int(alpha * 255)
    overlay_color_rgba = color_rgb + (alpha_int,)

    colored_mask_layer_np = np.zeros((height, width, 4), dtype=np.uint8)
    mask_logical = mask > 127
    colored_mask_layer_np[mask_logical] = overlay_color_rgba

    colored_mask_layer_pil = Image.fromarray(colored_mask_layer_np, "RGBA")
    result_img = Image.alpha_composite(img_rgba, colored_mask_layer_pil)
    return result_img


class GeminiChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemini Image Q&A Chat")

        # --- init Gemini client ---
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            messagebox.showerror(
                "API Key Error", "Please set GEMINI_API_KEY environment variable"
            )
            root.destroy()
            return
        self.client = genai.Client(api_key=api_key)

        # --- internal state ---
        self.rgb_image_bytes = None
        self.depth_image = None
        self.rgb_image_tk = None
        self.depth_image_tk = None
        self.original_pil_rgb = None

        # --- top panels: RGB (left) and Depth (right) ---
        images_frame = tk.Frame(root, height=300)
        images_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # RGB panel
        rgb_frame = tk.Frame(images_frame)
        rgb_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.rgb_label = tk.Label(
            rgb_frame, text="RGB Image", font=("Arial", 14, "bold")
        )
        self.rgb_label.pack(anchor="n", pady=(0, 5))
        self.rgb_drop_label = tk.Label(
            rgb_frame,
            text="Drag & drop RGB image here or click to open",
            bg="lightgray",
            relief="sunken",
            width=40,
            height=15,
        )
        self.rgb_drop_label.pack(fill="both", expand=True)  # removed expand=True
        self.rgb_drop_label.bind("<Button-1>", self.open_rgb_file_dialog)
        self.rgb_drop_label.drop_target_register(DND_FILES)
        self.rgb_drop_label.dnd_bind("<<Drop>>", self.handle_rgb_drop)

        # Depth panel
        depth_frame = tk.Frame(images_frame)
        depth_frame.pack(side="right", fill="both", expand=True, padx=5)
        self.depth_label = tk.Label(
            depth_frame, text="Depth Image", font=("Arial", 14, "bold")
        )
        self.depth_label.pack(anchor="n", pady=(0, 5))
        self.depth_drop_label = tk.Label(
            depth_frame,
            text="Drag & drop Depth image here or click to open",
            bg="lightgray",
            relief="sunken",
            width=40,
            height=15,
        )
        self.depth_drop_label.pack(fill="both", expand=True)  # removed expand=True
        self.depth_drop_label.bind("<Button-1>", self.open_depth_file_dialog)
        self.depth_drop_label.drop_target_register(DND_FILES)
        self.depth_drop_label.dnd_bind("<<Drop>>", self.handle_depth_drop)

        # Intrinsics panel
        intrinsics_frame = tk.Frame(root)
        intrinsics_frame.pack(side="top", fill="x", padx=10, pady=(0, 10))
        self.intrinsics_label = tk.Label(
            intrinsics_frame,
            text="Drop Camera Intrinsics CSV here or click to open",
            bg="lightgray",
            relief="sunken",
            height=2,
        )
        self.intrinsics_label.pack(fill="x", expand=True)
        self.intrinsics_label.bind("<Button-1>", self.open_intrinsics_file_dialog)
        self.intrinsics_label.drop_target_register(DND_FILES)
        self.intrinsics_label.dnd_bind("<<Drop>>", self.handle_intrinsics_drop)

        self.camera_intrinsics = None  # initialize

        # --- model selector ---
        model_frame = tk.Frame(root)
        model_frame.pack(fill="x", padx=10)
        tk.Label(model_frame, text="Choose Gemini Model:").pack(side="left")
        self.model_var = tk.StringVar(value="gemini-2.5-pro-preview-05-06")
        models = list(MODELS_INFO.keys())
        self.model_menu = tk.OptionMenu(
            model_frame, self.model_var, *models, command=self.update_model_desc
        )
        self.model_menu.pack(side="left", padx=5)
        self.model_desc_label = tk.Label(
            root, text="", wraplength=700, justify="left", fg="gray"
        )
        self.model_desc_label.pack(fill="x", padx=10, pady=(0, 10))
        self.update_model_desc(self.model_var.get())

        # --- chat display ---
        self.chat_display = scrolledtext.ScrolledText(
            root, state="disabled", width=80, height=10, wrap=tk.WORD
        )
        # keep chat area at fixed height, do not expand
        self.chat_display.pack(side="top", fill="x", expand=False, padx=10, pady=(0, 5))
        self.chat_display.tag_config("user", foreground="blue")
        self.chat_display.tag_config("gemini", foreground="green")

        # --- input area ---
        input_frame = tk.Frame(root)
        input_frame.pack(side="top", fill="x", padx=10, pady=(0, 10))
        self.question_entry = tk.Entry(input_frame)
        self.question_entry.pack(side="left", fill="x", expand=True)
        self.question_entry.bind("<Return>", lambda e: self.ask_question())
        ask_button = tk.Button(input_frame, text="Ask", command=self.ask_question)
        ask_button.pack(side="left", padx=10)

    def open_intrinsics_file_dialog(self, event=None):
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")], title="Select Camera Intrinsics CSV"
        )
        if file_path:
            self.load_intrinsics(file_path)

    def handle_intrinsics_drop(self, event):
        files = self.root.splitlist(event.data)
        if files:
            self.load_intrinsics(files[0])

    def load_intrinsics(self, file_path):
        try:
            with open(file_path, "r") as f:
                reader = csv.reader(f, delimiter=",")  # Use comma delimiter
                matrix = []
                for row in reader:
                    # row is a list of strings like ['1320.5165', ' 0.0', ' 957.6541']
                    float_row = [
                        float(item.strip()) for item in row if item.strip() != ""
                    ]
                    matrix.append(float_row)
            intrinsics = np.array(matrix)
            if intrinsics.shape != (3, 3):
                raise ValueError("Intrinsics matrix must be 3x3.")
            self.camera_intrinsics = intrinsics
            self.intrinsics_label.config(text=f"Loaded Intrinsics: {file_path}")
            self.append_chat(f"Loaded camera intrinsics from {file_path}", user=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load intrinsics: {e}")

    def open_rgb_file_dialog(self, event=None):
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")],
            title="Select an RGB Image",
        )
        if file_path:
            self.load_rgb_image(file_path)

    def open_depth_file_dialog(self, event=None):
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.bmp *.tiff *.tif")],
            title="Select a Depth Image",
        )
        if file_path:
            self.load_depth_image(file_path)

    def handle_rgb_drop(self, event):
        files = self.root.splitlist(event.data)
        if files:
            self.load_rgb_image(files[0])

    def handle_depth_drop(self, event):
        files = self.root.splitlist(event.data)
        if files:
            self.load_depth_image(files[0])

    def load_rgb_image(self, file_path):
        try:
            with open(file_path, "rb") as f:
                self.rgb_image_bytes = f.read()
            self.original_pil_rgb = Image.open(
                io.BytesIO(self.rgb_image_bytes)
            ).convert("RGBA")
            display_img = self.original_pil_rgb.copy()
            display_img.thumbnail(
                (
                    self.rgb_drop_label.winfo_width() or 400,
                    self.rgb_drop_label.winfo_height() or 400,
                )
            )
            self.rgb_image_tk = ImageTk.PhotoImage(display_img)
            self.rgb_drop_label.config(image=self.rgb_image_tk, text="")
            self.append_chat(f"Loaded RGB image: {file_path}", user=True, display=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load RGB image: {e}")

    def load_depth_image(self, file_path):
        try:
            depth_img = Image.open(file_path)
            # keep original for computations (in mm)
            self.depth_image = depth_img

            # DEBUG: print min/max
            depth_np = np.array(depth_img)
            print(f"Depth image min: {depth_np.min()}, max: {depth_np.max()}")

            # create a plasma‐mapped display image
            vmin, vmax = depth_np.min(), depth_np.max()
            if vmax > vmin:
                norm = (depth_np - vmin) / (vmax - vmin)
            else:
                norm = np.zeros_like(depth_np, dtype=float)
            cmap = cm.get_cmap("plasma")
            colored = cmap(norm)  # H x W x 4 floats in [0,1]
            rgb_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
            display_img = Image.fromarray(rgb_uint8)

            # thumbnail to fit widget
            display_img.thumbnail(
                (
                    self.depth_drop_label.winfo_width() or 400,
                    self.depth_drop_label.winfo_height() or 400,
                )
            )
            self.depth_image_tk = ImageTk.PhotoImage(display_img)
            self.depth_drop_label.config(image=self.depth_image_tk, text="")

            self.append_chat(
                f"Loaded Depth image: {file_path}", user=True, display=True
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Depth image: {e}")

    def update_model_desc(self, selected_model):
        desc = MODELS_INFO.get(selected_model, {}).get(
            "desc", "No description available."
        )
        self.model_desc_label.config(
            text=f"{MODELS_INFO[selected_model]['name']}: {desc}"
        )

    def append_chat(self, message, user=False, display=False):
        """
        Appends a message to the chat display.

        Args:
            message (str): The message to display.
            user (bool): If True, message is from user; else from Gemini.
            display (bool): If True, display in GUI; else only print/log. Default is False.
        """
        tag = "user" if user else "gemini"
        if display:
            self.chat_display.config(state="normal")
            self.chat_display.insert(
                tk.END, f"{'You' if user else 'Gemini'}: {message}\n", tag
            )
            self.chat_display.config(state="disabled")
            self.chat_display.see(tk.END)
        print(f"{'You' if user else 'Gemini'}: {message}")

    def ask_question(self):
        question = self.question_entry.get().strip()
        if not question:
            return
        if self.rgb_image_bytes is None:
            messagebox.showwarning(
                "No Image", "Please load an RGB image before asking a question."
            )
            return

        self.append_chat(question, user=True, display=True)
        self.question_entry.delete(0, tk.END)

        threading.Thread(
            target=self.call_gemini_api, args=(question,), daemon=True
        ).start()

    def call_gemini_api(self, question):
        try:
            image_part = types.Part.from_bytes(
                data=self.rgb_image_bytes, mime_type="image/jpeg"
            )
            selected_model = self.model_var.get()

            self.append_chat(
                "Sending request to Gemini API...", user=False, display=True
            )
            start_time = time.time()
            system_prompt = (
                "You are a helpful mobility assistant for blind. "
                "Keep the description concise and relevant to the question. "
                "Don't exceed 2 sentences unless asked for detailed description. "
                "When asked about the scene describe the objects in the scene that might be blocking the way unless asked for all objects or specific objects."
                "if asked how far something is, where something ist, or its distance or clock orientation then generate segmentation mask of this object and or surface and \
                provide JSON with label as description and segmentation mask and say nothing else"
            )

            # if questions include "far", "distance", "clock orientation" then add to system prompt generate segme
            response = self.client.models.generate_content(
                model=selected_model,
                contents=[image_part, question],  # question is a string
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                    system_instruction=system_prompt,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_ONLY_HIGH",
                        )
                    ],
                ),
            )

            elapsed = time.time() - start_time

            self.append_chat(
                f"(Response time: {elapsed:.2f} sec)", user=False, display=True
            )

            answer = response.text

            if not isinstance(answer, str) or not answer.strip():
                self.append_chat(
                    "Gemini returned empty or no textual response.",
                    user=False,
                    display=True,
                )
                return

            # 1. Extract the JSON block (if present)
            json_match = re.search(
                r"```json\s*(\[[\s\S]*?\])\s*```", answer, re.MULTILINE
            )
            if json_match:
                json_str = json_match.group(1)

                # 2. Remove the fenced‐JSON from the full answer to get only the narrative
                narrative = answer.replace(f"```json\n{json_str}\n```", "").strip()

                # 3. Append the raw JSON (display=False, so it only prints to console)
                self.append_chat(json_str, user=False, display=False)

                # 4. Then append the remaining narrative (display=True, so it shows up in the GUI)
                if narrative:
                    self.append_chat(narrative, user=False, display=True)
            else:
                # No JSON block → just show everything as before
                self.append_chat(answer, user=False, display=True)

            self._handle_gemini_response(answer, question)
        except Exception as e:
            self.append_chat(f"Error calling Gemini API: {e}", user=False)
            # print traceback for debugging
            import traceback

            traceback.print_exc()

    def _handle_gemini_response(self, answer, question):
        json_str = None
        json_match = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", answer, re.MULTILINE)
        if json_match:
            json_str = json_match.group(1)

        if json_str:
            try:
                parsed = json.loads(json_str)
            except Exception as e:
                self.append_chat(f"Failed to parse JSON: {e}", user=False)
                return

            if len(parsed) > 0:
                self._process_parsed_json(parsed, json_str, question)
            else:
                self.append_chat("Empty JSON response.", user=False)
        else:
            self.append_chat("No JSON block found in response.", user=False)

    def _process_parsed_json(self, parsed, json_str, question):
        first_item = parsed[0]
        if "box_2d" in first_item and "mask" not in first_item:
            bboxes = [item for item in parsed if "box_2d" in item]
            self.visualize_bboxes(bboxes)
            self.append_chat(f"Visualized {len(bboxes)} bounding boxes.", user=False)
        elif "mask" in first_item:
            segmentation_masks = parse_segmentation_masks(
                json_str, self.original_pil_rgb.height, self.original_pil_rgb.width
            )
            # If question asks for distance and depth image loaded, calculate depth per mask
            for mask in segmentation_masks:
                hour, angle_deg, dist_mm = self.compute_3d_angle_and_distance(mask)
                if dist_mm is None:
                    self.append_chat(
                        f"Could not compute distance/angle for '{mask.label}'",
                        user=False,
                        display=True,
                    )
                else:
                    self.append_chat(
                        f"'{mask.label}' is approximately at {dist_mm/1000:.2f} m, "
                        f"clock direction {hour} o'clock",
                        user=False,
                        display=True,
                    )

            self.visualize_segmentation_masks(segmentation_masks)
            self.append_chat(
                f"Visualized {len(segmentation_masks)} segmentation masks.", user=False
            )
        elif "point" in first_item:
            self.visualize_points(parsed)
            self.append_chat(f"Visualized {len(parsed)} points.", user=False)
        else:
            self.append_chat("Unknown JSON response format.", user=False)

    def compute_3d_angle_and_distance(self, mask: SegmentationMask):
        if self.depth_image is None or self.camera_intrinsics is None:
            return None, None, None

        depth_np = np.array(self.depth_image)
        depth_h, depth_w = depth_np.shape[:2]

        W_rgb, H_rgb = (
            mask.mask.shape[1],
            mask.mask.shape[0],
        )  # mask size (usually 1920x1440)

        # Scale intrinsics
        scale_x = depth_w / W_rgb
        scale_y = depth_h / H_rgb

        K = self.camera_intrinsics.copy()
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Resize mask to depth size
        pil_mask = Image.fromarray(mask.mask)
        pil_mask = pil_mask.resize((depth_w, depth_h), Image.Resampling.NEAREST)
        mask_np = np.array(pil_mask)
        ys, xs = np.where(mask_np > 127)
        if len(xs) == 0:
            return None, None, None

        u_c = np.mean(xs)
        v_c = np.mean(ys)

        depths = depth_np[ys, xs]
        if depths.size == 0:
            return None, None, None
        d_c = np.mean(depths)

        X = (u_c - cx) * d_c / fx
        Y = (v_c - cy) * d_c / fy
        Z = d_c

        vec = np.array([X, Y, Z])
        norm_vec = np.linalg.norm(vec)
        if norm_vec == 0:
            return None, None, None

        # Horizontal angle (yaw) for clock direction
        angle_horiz_rad = np.arctan2(X, Z)
        angle_horiz_deg = np.degrees(angle_horiz_rad)

        shifted_angle = angle_horiz_deg % 360  # normalize between 0 and 360

        # Map angle to clock hour, with 0° = 12 o'clock
        hour_index = int((shifted_angle + 15) // 30) + 1  # +15 for rounding
        if hour_index > 12:
            hour_index -= 12

        return hour_index, angle_horiz_deg, Z

    def visualize_points(self, points_list):
        pil_img = self.original_pil_rgb.copy()
        draw = ImageDraw.Draw(pil_img)
        width, height = pil_img.size

        try:
            font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 14)
        except IOError:
            font = ImageFont.load_default()

        colors = [
            "red",
            "green",
            "blue",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "yellow",
            "pink",
            "brown",
        ]

        radius = 7

        for i, item in enumerate(points_list):
            point = item.get("point")
            label = item.get("label", "")

            if point and len(point) == 2:
                y, x = point
                abs_x = int(x / 1000 * width)
                abs_y = int(y / 1000 * height)

                color = colors[i % len(colors)]

                # Draw circle
                draw.ellipse(
                    [
                        (abs_x - radius, abs_y - radius),
                        (abs_x + radius, abs_y + radius),
                    ],
                    fill=color,
                    outline="black",
                    width=2,
                )

                # Draw label near point (right and slightly above)
                text_pos = (abs_x + radius + 3, abs_y - radius - 3)
                # Background rectangle behind text for readability
                try:
                    bbox = draw.textbbox(text_pos, label, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except AttributeError:
                    tw, th = draw.textsize(label, font=font)

                draw.rectangle(
                    [text_pos, (text_pos[0] + tw, text_pos[1] + th)], fill="white"
                )
                draw.text(text_pos, label, fill=color, font=font)

        self._update_display_image(pil_img)

    def visualize_bboxes(self, bboxes):
        pil_img = self.original_pil_rgb.copy()
        draw = ImageDraw.Draw(pil_img)
        width, height = pil_img.size

        try:
            font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 14)
        except IOError:
            font = ImageFont.load_default()

        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "pink",
            "purple",
            "cyan",
            "magenta",
        ]

        for i, item in enumerate(bboxes):
            box = item["box_2d"]  # [y1, x1, y2, x2] normalized
            label = item.get("label", "object")

            y1, x1, y2, x2 = box
            abs_x1 = int(x1 / 1000 * width)
            abs_y1 = int(y1 / 1000 * height)
            abs_x2 = int(x2 / 1000 * width)
            abs_y2 = int(y2 / 1000 * height)

            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1

            color = colors[i % len(colors)]
            draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline=color, width=3)

            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except AttributeError:
                tw, th = draw.textsize(label, font=font)

            draw.rectangle([abs_x1, abs_y1 - th, abs_x1 + tw, abs_y1], fill=color)
            draw.text((abs_x1, abs_y1 - th), label, fill="white", font=font)

        self._update_display_image(pil_img)

    def visualize_segmentation_masks(self, segmentation_masks):
        pil_img = self.original_pil_rgb.copy()
        for i, mask in enumerate(segmentation_masks):
            color = additional_colors[i % len(additional_colors)]
            pil_img = overlay_mask_on_img(pil_img, mask.mask, color)

        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 14)
        except IOError:
            font = ImageFont.load_default()

        for i, mask in enumerate(segmentation_masks):
            color = additional_colors[i % len(additional_colors)]
            draw.rectangle([mask.x0, mask.y0, mask.x1, mask.y1], outline=color, width=3)
            if mask.label:
                draw.text(
                    (mask.x0 + 8, mask.y0 - 20), mask.label, fill=color, font=font
                )

        self._update_display_image(pil_img)

    def _update_display_image(self, pil_img):
        display_img = pil_img.copy()
        display_img.thumbnail(
            (
                self.rgb_drop_label.winfo_width() or 400,
                self.rgb_drop_label.winfo_height() or 400,
            )
        )
        self.rgb_image_tk = ImageTk.PhotoImage(display_img)
        self.rgb_drop_label.config(image=self.rgb_image_tk, text="")


def main():
    root = TkinterDnD.Tk()
    root.geometry("900x750")
    app = GeminiChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
