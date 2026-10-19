"""Shared fixtures: tiny MovieLens-20M-shaped raw data written to a temp dir."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

RATINGS_CSV = """userId,movieId,rating,timestamp
1,10,4.0,100
1,20,5.0,200
1,30,3.0,300
1,40,4.5,400
1,50,4.0,500
1,60,4.0,600
2,10,4.0,110
2,20,4.0,120
2,40,4.0,130
2,50,2.0,140
2,60,4.5,150
2,70,5.0,160
3,10,5.0,210
3,20,4.0,220
3,40,4.0,230
3,60,4.0,240
3,70,3.5,250
3,80,4.0,260
4,10,4.0,310
4,20,4.0,320
4,40,4.0,330
4,60,4.0,340
4,80,4.0,350
5,10,4.0,410
5,20,4.0,420
5,40,4.0,430
5,60,4.0,440
5,80,4.0,440
6,90,5.0,500
6,10,4.0,510
"""

MOVIES_CSV = """movieId,title,genres
10,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
20,Jumanji (1995),Adventure|Children|Fantasy
30,"Blade Runner (Director's Cut) (1982) ",Action|Sci-Fi|Thriller
40,Babylon 5,Sci-Fi
50,Heat (1995),Action|Crime|Thriller
60,Doom (2005),(no genres listed)
70,"Misérables, Les (1995)",Drama|Musical
80,Toy Story 2 (1999),Adventure|Animation|Children|Comedy|Fantasy
90,Pulp Fiction (1994),Comedy|Crime|Drama|Thriller
"""

TAGS_CSV = """userId,movieId,tag,timestamp
1,10,pixar,1000
2,10,"animated, funny",1001
3,90,quentin tarantino,1002
"""


@pytest.fixture
def tiny_raw_dir(tmp_path: Path) -> Path:
    """A directory that looks like data/raw/ml-20m with a handful of rows."""
    raw = tmp_path / "ml-20m"
    raw.mkdir()
    (raw / "ratings.csv").write_text(RATINGS_CSV)
    (raw / "movies.csv").write_text(MOVIES_CSV)
    (raw / "tags.csv").write_text(TAGS_CSV)
    return raw


@pytest.fixture
def interactions() -> pl.DataFrame:
    """Remapped positive interactions for two users, already in the split-input schema."""
    return pl.DataFrame(
        {
            "user_idx": [0, 0, 0, 0, 1, 1, 1],
            "item_idx": [0, 1, 2, 3, 0, 1, 2],
            "rating": [4.0, 5.0, 4.5, 4.0, 4.0, 4.0, 5.0],
            "timestamp": [100, 200, 300, 400, 10, 30, 20],
        },
        schema={
            "user_idx": pl.Int32,
            "item_idx": pl.Int32,
            "rating": pl.Float32,
            "timestamp": pl.Int64,
        },
    )
