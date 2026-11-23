
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from enroll import MIN_PHOTOS, enroll_user
from verify import verify_user


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit


@app.route("/")
def index():
    return render_template("index.html", min_photos=MIN_PHOTOS)


@app.route("/enroll", methods=["POST"])
def enroll():
    name = request.form.get("name", "")
    files = request.files.getlist("photos")

    try:
        images = [Image.open(file.stream) for file in files if file.filename]
        success, messages = enroll_user(name, images)
    except UnidentifiedImageError:
        success, messages = False, ["Error: one of the files is not an image."]
    except (ValueError, OSError) as error:
        success, messages = False, [f"Error: {error}"]

    return render_template(
        "result.html", title="Enrollment", success=success, messages=messages
    )


@app.route("/verify", methods=["POST"])
def verify():
    name = request.form.get("name", "")
    file = request.files.get("photo")

    try:
        if file is None or not file.filename:
            raise ValueError("please upload a photo")

        image = Image.open(file.stream)
        success, messages = verify_user(name, image)
    except UnidentifiedImageError:
        success, messages = False, ["Error: that file is not an image."]
    except (ValueError, OSError) as error:
        success, messages = False, [f"Error: {error}"]

    return render_template(
        "result.html", title="Login", success=success, messages=messages
    )


if __name__ == "__main__":
    app.run(debug=False)
