import click
import os
import pathlib


@click.command(name="capture", help="Captures images from web camera")
@click.option('-r', '--rotate', is_flag=True, help="Rotate the web camera 90 degrees (default: False)")
@click.option('-s', '--source', default=0, type=int, help="Web camera source (default: 0)")
def capture(rotate: bool, source: int):
    import cv2

    click.echo("Capturing images from web camera...")
    click.echo("Rotate: {}".format(rotate))
    click.echo("Source: {}".format(source))

    click.echo("Press 'q' to quit the capture.")
    click.echo("Press the space bar to save the image.")

    output_folder = pathlib.Path(os.getcwd()) / 'captures'
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    cam = cv2.VideoCapture(source)

    while True:
        ret, frame = cam.read()

        if not ret:
            break

        # Rotate the frame 90 degrees if specified
        if rotate:
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated_frame = frame

        # Display the rotated frame
        cv2.imshow('Camera', rotated_frame)

        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            # Press 'q' to exit the loop
            break
        elif key == ord(' '):
            # Press space bar to save the frame
            image_filename = os.path.join(output_folder, f"capture_{cv2.getTickCount()}.jpg")
            cv2.imwrite(image_filename, rotated_frame)
            click.echo(f"Saved: {image_filename}")

    # Release the capture object
    cam.release()
    cv2.destroyAllWindows()


@click.command(name="train", help="Trains a YOLO model based on the captured and labeled images")
@click.option('-e', '--epochs', default=100, type=int, help="Number of epochs to train the model (default: 100)")
@click.option('-m', '--model-name', default='yolo11n', type=str, help="Name of the model to train (default: yolo11n)")
@click.option('--device', default='cpu', type=str, help="Device to use for training (default: cpu)")
@click.option('-d', '--data', default='data/cubes_on_desk_dataset.yaml', type=str, help="Dataset YAML to train on (default: data/cubes_on_desk_dataset.yaml)")
def train(epochs: int, model_name: str, device: str, data: str):
    from ultralytics import YOLO

    click.echo("Training the model...")
    click.echo("Model: {}".format(model_name))
    click.echo("Epochs: {}".format(epochs))

    # Load a model
    model = YOLO(f"{model_name}.yaml")  # build a new model from YAML
    model = YOLO(f"{model_name}.pt")  # load a pretrained model (recommended for training)
    model = YOLO(f"{model_name}.yaml").load(f"{model_name}.pt")  # build from YAML and transfer weights


    dataset_path = str((pathlib.Path(os.getcwd()) / data).resolve())
    if not os.path.exists(dataset_path):
        click.echo(f"Dataset path {dataset_path} does not exist.")
        return

    # Train the model
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=640,
        device=device
    )


@click.group()
def cli():
    pass


cli.add_command(capture)
cli.add_command(train)


if __name__ == '__main__':
    cli()
