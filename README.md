![Double pendulum](double_pendulum.png)


The code behind [@pendulum_bot](https://twitter.com/pendulum_bot) Twitter bot which posts animations of a double pendulum released from a random position to swing for 30 seconds.


&nbsp;


# Basic usage

To create an animation of a random double pendulum:

```python
>>> from simulation import create_random_example, simulate
>>> from animations import single_animation
>>> rand_ex = create_random_example()
>>> results = simulate(rand_ex)
>>> single_animation(results, rand_ex)
```

The animation is saved as .mp4 video in `animations` subdirectory.

---

To create an animation and post it on Twitter, a valid API key is needed, and should be stored in `api_key.txt`.

```python
>>> from tweet_it import new_tweet
>>> new_tweet() # creates a new animation of random double pendulum
# or
>>> new_tweet('existing_file', 'My custom Twitter status')
```

---

To create double pendulum with the exact values for initial conditions:

```python
>>> from pendulum import Pendulum, DoublePendulum
>>> p1 = Pendulum(m=2.7, x=2.5, y=3.7, u=0, v=0)
>>> p2 = Pendulum(m=3.1, x=0.2, y=6.3, u=0, v=0)
>>> dp = DoublePendulum(p1, p2)
```

---

To create multiple pendulums with slight perturbations of initial conditions to observe chaotic behaviour:

```python
>>> from simulation import create_random_example, create_perturbations, simulate_multiple_examples
>>> from animations import multi_animation
>>> rand_ex = create_random_example()
>>> perturbed = create_perturbations(10, rand_ex, amount=1e-5)
>>> results = simulate_multiple_examples(perturbed)
>>> multi_animation(results, rand_ex)
```


&nbsp;


# Installation

```
git clone https://github.com/narimiran/double_pendulum.git
cd double_pendulum
```

### Dependencies

* Python 3
* numpy (running simulations)
* matplotlib (creating animations)
* ffmpeg or avconv/libavtools (saving videos)
* twython (posting Twitter updates)


&nbsp;


# FAQ

Q: *Why do you use Cartesian coordinates? I prefer polar coordinates.*

A: The initial task I was given was to implement double pendulum as DAE system in Cartesian coordinates. The idea for animations and Twitter bot came later, and Cartesian coordinates remained.

Q: *Which Runge-Kutta methods can I use?*

A: Any of these:

* Forward Euler (`Euler`)
* Explicit midpoint (`ExplicitMidpoint`)
* Ralston's method (`Ralston`)
* Kutta's 3rd order method (`Kutta3`)
* *the* Runge-Kutta 4th order method (`RK4`)
* Runge-Kutta-Fehlberg (`RKF`)
* Cash-Karp (`Cash-Karp`)
* Dormand-Prince method (`DOPRI5`)

Q: *Why can't I use implicit Runge-Kutta methods?*

A: Implicit methods require different solving method (solving a system of non-linear equations). This is not (yet) implemented.

Q: *Is there any damping/friction?*

There is no damping and no friction. The only force acting on the system is gravity.

Q: *Couldn't all/some of this be done simpler?*

A: Probably.


&nbsp;


# License

[MIT License](LICENSE.txt)