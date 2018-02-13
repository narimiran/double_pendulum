from twython import Twython
from simulation import create_random_example, simulate
from animations import create_comment, single_animation


def create_content():
    """Creates an animation of a random double pendulum example.

    Returns:
        tuple (string, string):
            - name of the file of a .mp4 video (without extension), created
                in ./animations subdirectory
            - string with initial conditions which will be posted as
                Twitter status
    """
    while True:
        rs = create_random_example()
        try:
            r = simulate(rs, step_size=0.004)
            break
        except FloatingPointError:
            continue
    filename = single_animation(r, rs)
    comment = create_comment(rs)
    return filename, comment


def new_tweet(filename=None, status=None):
    """Posts a new tweet.

    Status are the initial conditions, video is attached.
    To successfully post, a valid API key is needed to be stored in `api_key.txt`.

    Args:
        filename (string): name of the file containing video (without extension)
        status (string): text which will be posted as Twitter status
    """
    if filename is None:
        filename, comment = create_content()
        if status is None:
            status = comment

    with open("api_key.txt") as f:
        api_data = f.readline().split(';')
    twitter = Twython(*api_data)

    video = open('./animations/{}.mp4'.format(filename), 'rb')
    response = twitter.upload_video(media=video, media_type='video/mp4')
    twitter.update_status(status=status, media_ids=[response['media_id']])


if __name__ == '__main__':
    new_tweet()
