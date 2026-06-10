import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from modoc_pipeline import bgm_client


class BgmClientTests(unittest.TestCase):
    def test_generate_all_bgm_writes_wav_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            def fake_generate(*, language, output_path, api_key, **kwargs):
                bgm_client.write_wave(output_path, b"\x00\x00" * 16)
                return output_path

            with patch.dict("os.environ", {"MODOC_BGM_SOURCE": "lyria"}), patch.object(bgm_client, "generate_meme_bgm", side_effect=fake_generate):
                result = bgm_client.generate_all_bgm(
                    output_dir=output_dir,
                    api_key="unused",
                    languages={"english"},
                )

            self.assertEqual(set(result), {"english"})
            self.assertEqual(result["english"].suffix, ".wav")
            self.assertTrue(result["english"].exists())

    def test_write_wave_uses_lyria_audio_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bgm.wav"
            bgm_client.write_wave(path, b"\x00\x00" * 32)

            with wave.open(str(path), "rb") as handle:
                self.assertEqual(handle.getnchannels(), bgm_client.LYRIA_CHANNELS)
                self.assertEqual(handle.getframerate(), bgm_client.LYRIA_SAMPLE_RATE)
                self.assertEqual(handle.getsampwidth(), bgm_client.LYRIA_SAMPLE_WIDTH)

    def test_default_model_uses_live_music_endpoint_model(self):
        self.assertEqual(bgm_client.BgmConfig().model, "models/lyria-realtime-exp")

    def test_generate_all_bgm_uses_plan_prompt_and_config_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            calls = []

            def fake_generate(**kwargs):
                calls.append(kwargs)
                bgm_client.write_wave(kwargs["output_path"], b"\x00\x00" * 16)
                return kwargs["output_path"]

            meme_plan = {
                "english": {
                    "bgm_prompt": "Bright indie pop at 118 BPM with soft percussion.",
                    "bgm_config": {"bpm": 118, "density": 0.5, "brightness": 0.8},
                }
            }
            with patch.dict("os.environ", {"MODOC_BGM_SOURCE": "lyria"}), patch.object(bgm_client, "generate_meme_bgm", side_effect=fake_generate):
                result = bgm_client.generate_all_bgm(
                    output_dir=output_dir,
                    api_key="unused",
                    meme_plan=meme_plan,
                    languages={"english"},
                )

            self.assertEqual(set(result), {"english"})
            self.assertEqual(calls[0]["prompt_override"], meme_plan["english"]["bgm_prompt"])
            self.assertEqual(calls[0]["music_config_override"], meme_plan["english"]["bgm_config"])

    def test_bgm_prompt_guardrail_adds_no_vocals(self):
        prompt = bgm_client._with_instrumental_guardrail("Soft upbeat piano")
        self.assertIn("no vocals", prompt.lower())

    def test_copy_random_stock_bgm_writes_mp3(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            stock_dir = base / "stock"
            stock_dir.mkdir()
            source = stock_dir / "track.mp3"
            source.write_bytes(b"fake mp3 bytes")
            output = base / "english_bgm.mp3"

            result = bgm_client._copy_random_stock_bgm(output, stock_dir=stock_dir)

            self.assertEqual(result, output)
            self.assertEqual(output.read_bytes(), b"fake mp3 bytes")


if __name__ == "__main__":
    unittest.main()
