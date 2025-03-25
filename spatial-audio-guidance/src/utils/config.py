# Contents of /spatial-audio-guidance/spatial-audio-guidance/src/utils/config.py

class Config:
    def __init__(self):
        self.audio_settings = {
            "volume": 0.5,
            "sample_rate": 44100,
            "channels": 2
        }
        self.scene_settings = {
            "depth_estimation_method": "stereo",
            "object_detection_threshold": 0.5
        }
        self.guidance_settings = {
            "tts_engine": "default",
            "audio_feedback_enabled": True
        }

    def get_audio_settings(self):
        return self.audio_settings

    def get_scene_settings(self):
        return self.scene_settings

    def get_guidance_settings(self):
        return self.guidance_settings

    def update_audio_setting(self, key, value):
        if key in self.audio_settings:
            self.audio_settings[key] = value

    def update_scene_setting(self, key, value):
        if key in self.scene_settings:
            self.scene_settings[key] = value

    def update_guidance_setting(self, key, value):
        if key in self.guidance_settings:
            self.guidance_settings[key] = value