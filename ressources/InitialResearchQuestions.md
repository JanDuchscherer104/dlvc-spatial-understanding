## Research Questions

In the context of this project:

```
3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users
Problem: scene description methods based on color images lack precise spatial
information, including accurate distances, orientations, and relative positioning of obstacles,
landmarks, and pathways. This limitation prevents blind and visually impaired individuals
from safely navigating complex indoor and outdoor spaces independently and confidently.
Goal: Develop and evaluate a 3D spatial scene description system optimized for blind and
visually impaired smartphone users. The solution must describe distances, orientations, and
spatial relationships of key elements within the user's environment and offer clear guidance
to safely avoid obstacles or reach specific destinations.
Tasks:
• Implement an application (first on laptop, then on iPhone) that leverages Gemini
Spatial Understanding capabilities to provide precise, interactive 3D spatial scene
descriptions tailored for blind users.
• Evaluate performance in diverse scenarios, including sidewalks, unexpected
barriers and obstacles, public transport, grocery shopping, crossing streets,
finding objects and entrances, etc.
• Quantitatively assess system effectiveness in real-world situations using
smartphone-captured imagery and depth sensors (e.g., iPhone Pro).
• Identify and document performance in specific challenging scenarios Document
detailed evaluation metrics, including accuracy of 3D descriptions, directional
guidance, per-scenario errors, qualitative results, and response time.
Resources:
• Gemini Spatial Understanding Video Overview: https://www.youtube.com/watch?v=-
XmoDzDMqj4
• Gemini Spatial Understanding Notebook: https://github.com/google-
gemini/cookbook/blob/main/examples/Spatial_understanding_3d.ipynb
• Gemini Cookbook: https://github.com/google-gemini/cookbook
• Further approaches for 3D scene understanding:
https://github.com/ActiveVisionLab/Awesome-LLM-3D
Bonus Task:
• Option 1: Fine-tune Gemini Spatial Understanding models specifically for blind
navigation scenarios to increase accuracy and reduce errors.
• Option 2: Explore methods and approaches to make this an on-device solution
instead of cloud-based solution. What components are needed in such a pipeline
(e.g., object detection / segmentation model, depth data of the objects, VLM, LLM,
etc.)?
Contact:
```

- What state of the art zero-shot models are there like Gemini 2.0 flash spatial understanding that are capable of providing generic detections or segmentations given a user prompt?
- Are there models that can directly work with the 3D data (as well as other modalities that are yielded by the iPhone, eg ego poses)
- Are these models capable of also providing:
  - SLAM integration?
  - Produce a dynamic scene graph (operate on the temporal sequence of data points)?
