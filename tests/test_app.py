from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_renders_upload_without_import_errors():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    app.run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "AI Volleyball Back-View Team Analyzer"
    assert len(app.get("file_uploader")) == 1
