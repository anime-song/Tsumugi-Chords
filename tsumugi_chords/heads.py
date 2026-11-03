"""Shared sizes of the chord outputs (the subset of SheetSage2-Refined's heads module the models use)."""

FRAME_RATE = 25  # output frames per second
DETAIL = 745  # root x quality in chord_detail order (12 * quality + root) and N last
KEYS = 88  # notes A0..C8
BINARY = ("chord_boundary", "key_boundary", "beat", "downbeat", "group")
