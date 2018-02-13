from datetime import datetime
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation


def create_metadata(title, comment='', author=None):
    """Creates metadata for video.

    Args:
        title (string): title of the video
        comment (string): comment of the video
        author (strin): author of the video

    Returns:
        dict
    """
    author = 'https://github.com/narimiran' if author is None else author
    return dict(artist=author, title=title, comment=comment)


def create_comment(example):
    """Creates a comment for video, by listing the initial values of the
    example `example`.

    Args:
        example (`DoublePendulum` class instance)

    Returns:
        string
    """
    status = ['Initial conditions:']
    for i, b in enumerate((example._b1, example._b2), 1):
        s = 'Pendulum {i}: position: ({b.x:.2f}, {b.y:.2f}), mass: {b.m:.2f}'.format(i=i, b=b)
        status.append(s)
    return '\n'.join(status)


def single_animation(system, ex, fig_size=(8, 8), hide_axes=True, filename=None):
    """Creates and saves an animation of a single example/system.

    The animation is saved in `./animations` subdirectory.
    Animations are done in 50 fps.
    Bob size is proportional to bob mass.
    Trailing dots show previous pendulum positions.

    Args:
        system (`ERK_DAE1` class instance): DAE system with the results in
            `ys` and `zs` attributes.
        ex (`DoublePendulum` class instance)
        fig_size (tuple (float, float)): size of the figure
        hide_axes (bool): should axis label be hidden of visible. Hidden is
            the default, as it makes animation creation much faster
        filename (string): name of the file (without an extension) where
            animation should be saved

    Returns:
        filename (string)
    """
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(
        xlim=(system.ys[:, 1].min() * 1.2, system.ys[:, 1].max() * 1.2),
        ylim=(system.ys[:, 3].min() * 1.2, max(0.2, system.ys[:, (2, 3)].max()) * 1.2)
    )
    if hide_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')

    ax.plot(0, 0, 'o', ms=8, c='C0')
    line, = ax.plot([], [], '-', lw=3, c='C0')
    m1, = ax.plot([], [], 'o', ms=4*ex.M[0, 0], c='C1')
    m2, = ax.plot([], [], 'o', ms=4*ex.M[1, 1], c='C2')
    p1, = ax.plot([], [], 'o', ms=1, c='C1')
    p2, = ax.plot([], [], 'o', ms=1, c='C2')

    fig.tight_layout()

    skip = int(0.02 / system.h) # 50fps --> 0.02 sec per frame

    def animate(i):
        i *= skip
        line.set_data(
            [0, system.ys[i, 0], system.ys[i, 1]],
            [0, system.ys[i, 2], system.ys[i, 3]])
        p1.set_data(
            system.ys[0:i:skip, 0],
            system.ys[0:i:skip, 2])
        p2.set_data(
            system.ys[0:i:skip, 1],
            system.ys[0:i:skip, 3])
        m1.set_data(
            system.ys[i, 0],
            system.ys[i, 2])
        m2.set_data(
            system.ys[i, 1],
            system.ys[i, 3])
        return line, p1, p2, m1, m2

    anim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        frames=len(system.ys)//skip,
        interval=20,
        blit=True,
    )

    filename = datetime.now().strftime('%Y-%m-%d_%H-%M') if filename is None else filename
    metadata = create_metadata(filename, comment=create_comment(ex))

    if not os.path.isdir(os.path.join(os.getcwd(), 'animations')):
        os.mkdir('animations')

    try:
        writer = animation.AVConvWriter(fps=50, bitrate=-1, metadata=metadata)
        anim.save('./animations/{}.mp4'.format(filename), writer=writer)
    except FileNotFoundError:
        writer = animation.FFMpegWriter(fps=50, bitrate=-1, metadata=metadata)
        anim.save('./animations/{}.mp4'.format(filename), writer=writer)

    return filename


def multi_animation(systems, ex, fig_size=(8, 8), hide_axes=True, filename=None):
    """Creates and saves an animation of a multiple systems.

    The animation is saved in `./animations` subdirectory.
    beforehand.
    Animations are done in 50 fps.
    Bob size is proportional to bob mass.
    Trailing dots show previous pendulum positions.

    Args:
        systems (list of `ERK_DAE1` class instance): list of DAE systems with
            the results in `ys` and `zs` attributes.
        ex (`DoublePendulum` class instance)
        fig_size (tuple (float, float)): size of the figure
        hide_axes (bool): should axis label be hidden of visible. Hidden is
            the default, as it makes animation creation much faster
        filename (string): name of the file (without an extension) where
            animation should be saved

    Returns:
        filename (string)
    """
    ys = np.array([s.ys for s in systems])

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(
        xlim=(ys[:, :, 1].min() * 1.2, ys[:, :, 1].max() * 1.2),
        ylim=(ys[:, :, 3].min() * 1.2, max(0.2, ys[:, :, (2, 3)].max()) * 1.2)
    )
    if hide_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')

    ax.plot(0, 0, 'o', ms=8, c='k')

    lines, m1s, m2s, trails = [], [], [], []
    for k in range(len(systems)):
        lines.append(ax.plot([], [], '-', lw=3, c='C{}'.format(k))[0])
        m1s.append(ax.plot([], [], 'o', ms=4*ex.M[0, 0], c='C{}'.format(k))[0])
        m2s.append(ax.plot([], [], 'o', ms=4*ex.M[1, 1], c='C{}'.format(k))[0])
        trails.append(ax.plot([], [], 'o', ms=1, c='C{}'.format(k))[0])

    fig.tight_layout()

    skip = int(0.02 / systems[0].h) # 50fps --> 0.02 sec per frame

    def animate(i):
        i *= skip
        for k in range(len(systems)):
            lines[k].set_data(
                [0, ys[k, i, 0], ys[k, i, 1]],
                [0, ys[k, i, 2], ys[k, i, 3]])
            m1s[k].set_data(
                ys[k, i, 0],
                ys[k, i, 2])
            m2s[k].set_data(
                ys[k, i, 1],
                ys[k, i, 3])
            trails[k].set_data(
                ys[k, max(0, i-100*skip):i:skip, 1],
                ys[k, max(0, i-100*skip):i:skip, 3])
        return lines, m1s, m2s, trails

    anim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        frames=ys.shape[1]//skip,
        interval=20,
        blit=False,
    )

    filename = datetime.now().strftime('%Y-%m-%d_%H-%M') if filename is None else filename
    metadata = create_metadata(filename, comment=create_comment(ex))

    if not os.path.isdir(os.path.join(os.getcwd(), 'animations')):
        os.mkdir('animations')

    try:
        writer = animation.AVConvWriter(fps=50, bitrate=-1, metadata=metadata)
        anim.save('./animations/{}.mp4'.format(filename), writer=writer)
    except FileNotFoundError:
        writer = animation.FFMpegWriter(fps=50, bitrate=-1, metadata=metadata)
        anim.save('./animations/{}.mp4'.format(filename), writer=writer)

    return filename


def __run_basic_example():
    from simulation import *
    while True:
        rs = create_random_example()
        try:
            r = simulate(rs, method=Euler, step_size=0.005)
            break
        except FloatingPointError:
            continue
    print(rs._b1)
    print(r.ys[:3])
    print(create_metadata(''))
    single_animation(r, rs)

    mtd = [Euler, DOPRI5, ExplicitMidpoint]
    mms = simulate_multiple_methods(rs, mtd, duration=20, step_size=0.01)
    multi_animation(mms, rs)

    exes = create_perturbations(5, rs, amount=1e-2)
    ps = simulate_multiple_examples(exes, RK4, 30, 0.005)
    multi_animation(ps, rs)


if __name__ == '__main__':
    __run_basic_example()
