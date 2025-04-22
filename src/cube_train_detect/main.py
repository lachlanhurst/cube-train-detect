import click
import cv2
import os
import pathlib


@click.command(name="capture", help="Captures images from web camera")
@click.option('-r', '--rotate', is_flag=True, help="Rotate the web camera 90 degrees (default: False)")
@click.option('-s', '--source', default=0, type=int, help="Web camera source (default: 0)")
def capture(rotate: bool, source: int):
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


@click.group()
def cli():
    pass


cli.add_command(capture)


if __name__ == '__main__':
    cli()
