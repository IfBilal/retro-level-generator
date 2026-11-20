"""The ONE tile table (ledger F6). Everything else imports from here; never retype tile ids.

Explain-me:
  What: tile ids 0..12 with their characters, names, colours and the solid/passable/enemy/pipe/cannon groups.
  In/Out: constants only. Used by the detector, renderer, dataset, playability checker and game export.
  Try: add a tile char here and run tests/test_data.py -> the smb.json consistency test fails (drill 2).
"""
from __future__ import annotations

# Order is authoritative (ledger F6): ids follow the sorted keys of smb.json.
CHARS = ('-', '<', '>', '?', 'B', 'E', 'Q', 'S', 'X', '[', ']', 'b', 'o')
NAMES = ('sky', 'pipe_top_left', 'pipe_top_right', 'question_full', 'cannon_top', 'enemy',
         'question_empty', 'breakable', 'ground', 'pipe_body_left', 'pipe_body_right',
         'cannon_support', 'coin')
SOLID = frozenset({1, 2, 3, 4, 6, 7, 8, 9, 10, 11})
PASSABLE = frozenset({0, 12})
ENEMY = frozenset({5})
PIPE = frozenset({1, 2, 9, 10})
CANNON = frozenset({4, 11})
N_TILES = 13
CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}

# Named ids so detector code reads like English.
SKY, PIPE_TL, PIPE_TR, QBLOCK, CANNON_TOP, ENEMY_ID, QEMPTY, BREAKABLE, GROUND, PIPE_BL, PIPE_BR, \
    CANNON_SUP, COIN = range(13)

# Placeholder colours for debug renders (doc 11 §1); real sprites come from the asset pipeline (doc 26).
COLOURS = ('#9ccfff', '#20c040', '#20c040', '#f8d030', '#303030', '#e03030', '#a08060', '#c86432',
           '#a0522d', '#109030', '#109030', '#606060', '#ffd700')
