# IN BLOOM - Rose

A single line of light grows into a stem, leaves unfurl, a bud forms and then,
petal by petal, a red rose opens. It sways gently, dewdrops gather on the petals and
fall with a little splash. It is drawn only with letters, numbers and symbols.

![IN BLOOM](images/preview.png)

## How to run

```bash
pip install numpy pygame-ce
python rose.py
```

Save one frame without opening a window (here at t = 16 s):

```bash
python rose.py --shot 16 rose.png
```

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `R` | Restart |
| `S` | Save a screenshot |
| `ESC` | Quit |

`rose.py` is stand-alone: it does not need `ascii_engine.py`.

## What happens

| Time | Moment |
|---|---|
| 0 - 4 s | A travelling point of light draws the stem upward and the leaves unfurl |
| 4 - 6.5 s | Sepals open and a closed, egg-shaped bud fills in |
| 6.5 - 14 s | The bloom opens, outer petals first |
| 14 s + | The rose sways in the breeze and dewdrops fall and splash |

## Customise

At the top of `rose.py`:

- `HEAD_SCALE`, `STEM_TOP`: size of the flower and height of the stem
- `T_STEM`, `T_BLOOM0`, `T_BLOOM1`: timing of growth and bloom
- `PETAL_COL` (further down): petal colours, from the centre to the tips
- `LEAVES`: where the leaves sit, their angle, length and width
- `RAMPS`: the characters used for shading, from dark to bright

## How it works

- The flower is **one parametric spiral surface** that loosens from a bud into a full rose.
- The stem is a tube along a curve, and the leaves are folded, drooping surfaces.
- Points are lit (diffuse, specular and rim light), projected with perspective and
  depth-sorted into a grid of character cells.
- Brightness picks the character, and depth jumps become `/ \ | _ -` outlines.
- Particles add sparks, dewdrops and splashes, with a glow, twinkling stars and a faint reflection.

*Inspired by "IN BLOOM" by [JIN2 (@jin_22ee)](https://instagram.com/jin_22ee). This is an
independent implementation, not the artist's original code.*

---

Made with ♥ by **Aavi** · [GitHub @awnish9002](https://github.com/awnish9002) ·
[Instagram @code_with_aavi](https://www.instagram.com/code_with_aavi/) ·
[All pieces](../README.md)
