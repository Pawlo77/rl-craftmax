"""Tests for the Pillow text overlays."""

import numpy as np

from wut_murl_craftax.visualisation import text


def test_title_card_is_opaque_and_sized() -> None:
    card = text.title_card((640, 360), "Craftax agent highlights", "test cut")
    assert card.size == (640, 360)
    assert card.mode == "RGB"
    pixels = np.asarray(card)
    assert (pixels != pixels[0, 0]).any()  # text was drawn


def test_caption_panel_is_transparent_outside_its_box() -> None:
    panel = text.caption_panel(
        (1920, 1080), (40, 760, 420), "Collect iron", "Craftax · steps 1-20"
    )
    alpha = np.asarray(panel)[..., 3]
    assert alpha[:700].max() == 0
    assert alpha[760:800, 40:400].max() > 0


def test_banner_is_centred_on_the_given_point() -> None:
    img = text.banner((1920, 1080), (960, 300), "UNLOCKED", "Collect Iron")
    ys, xs = np.nonzero(np.asarray(img)[..., 3])
    assert abs((xs.min() + xs.max()) / 2 - 960) <= 2
    assert ys.min() < 300 < ys.max()


def test_wrap_respects_width() -> None:
    f = text.font(40)
    lines = text.wrap("defeat the zombie after a very long fight", f, 200)
    assert len(lines) > 1
    assert all(f.getlength(line) <= 200 or " " not in line for line in lines)


def test_pretty_turns_enum_names_into_words() -> None:
    assert text.pretty("MAKE_STONE_PICKAXE") == "Make Stone Pickaxe"
    assert text.pretty("4 enemies at once") == "4 enemies at once"


def test_label_bar_and_strip_sizes() -> None:
    assert text.label_bar("seed 3", 352).size == (352, 40)
    assert text.text_strip("Unlocks iron", 528).size == (528, 34)
