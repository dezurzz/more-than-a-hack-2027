"""Bagian 0 -- every constant and threshold the rule-based pipeline uses, in one place.

None of these numbers has been tuned. They are the starting points from DZAKY APPROACH.md: the
[ASLI] ones come from Vidal-Codina et al. (2022), the [ADAPTASI] ones are ours. Tahap 3 picks the
final values on external games (Sportec Open DFL / Metrica sample data), never on the public
ground truth -- RULES.md forbids tuning on it, and EVALUATION.md shows it picks the wrong answer
even when nobody is trying to.

Keeping them in one frozen dataclass means a tuning run can override any of them without touching
the logic, and a run is fully described by the Config it was given:

    from mtah.members.dzaky11.config import Config
    cfg = Config(R_PZ=2.0, AMBANG_MATI=1.25)

Names follow the pseudocode so the document and the code can be read side by side.
"""

import dataclasses
from mtah.config import ROOT  # noqa: F401  (also puts the repository root on sys.path, for games)
from games import GAME, GAMES  # noqa: E402

# Team codes, shared by every module. Same convention as starter/baseline.py: the first team in
# the ATD metadata is home.
H, A, NONE = 0, 1, -1


@dataclasses.dataclass(frozen=True)
class Config:
    FPS: int = GAMES[GAME]["fps"]
    # Metres, measured from the corrected tracking (games.py); the ATD metadata says 105 x 68.
    LAPANGAN_P: float = GAMES[GAME]["pitch"][0]
    LAPANGAN_L: float = GAMES[GAME]["pitch"][1]

    # -- Bagian 1: pre-processing [ADAPTASI] -----------------------------------------------------
    # Ball gaps up to this many frames (~0.5 s) are interpolated; longer ones stay missing,
    # because on this data a missing ball is not a dead ball.
    GAP_INTERPOLASI: int = 12
    # [ASLI] Savitzky-Golay smoothing of the ball track, applied per visible segment.
    SG_JENDELA: int = 7
    SG_ORDE: int = 2

    # -- Bagian 2: control frames [ASLI] ---------------------------------------------------------
    # Paper uses 0.5-1 m on multi-camera tracking; widened for one tactical camera.
    R_PZ: float = 1.5   # possession zone radius around each player (m)
    R_DZ: float = 1.5   # duel zone radius around the ball (m), >= R_PZ

    # -- Bagian 3: touches [ASLI] -----------------------------------------------------------------
    EPS_THETA: float = 0.9   # cosine between incoming and outgoing direction; below = deflected
    EPS_V: float = 1.5       # ball speed change that counts as a touch (m/s)
    # [ADAPTASI] Below this speed the ball has no meaningful direction, so the turn test is skipped.
    V_ARAH_MIN: float = 1.0

    # -- Bagian 3: control [ADAPTASI] ------------------------------------------------------------
    # An opposing touch moves possession only if it is controlled: the ball stays with that team
    # for at least T_KONTROL frames, or that team touches it N_SENTUH_KONTROL times in a row.
    T_KONTROL: int = 12
    N_SENTUH_KONTROL: int = 2

    # -- Bagian 5: dead ball [ADAPTASI] ----------------------------------------------------------
    MARGIN_LUAR: float = 0.5          # metres beyond a line before the ball counts as out
    V_PEMAIN_DIAM: float = 1.0        # mean player speed (m/s) below which play looks stopped
    V_BOLA_DIAM: float = 0.5          # ball speed (m/s) below which the ball is at rest
    DURASI_DIAM_MIN: int = 2 * 25     # frames the ball must be seen at rest to count as placed

    # Gap rule. On the provided ATD the ball goes missing in long blocks -- 132 gaps of 5 s or
    # more covering ~41% of graded frames, almost none between 1 s and 5 s, and the ball unseen
    # at nearly every tagged corner and kickoff. A gap this long is taken to be a stoppage.
    # Because there are almost no gaps between 1 and 5 s, any value in that range gives the same
    # result; 5 s is the conservative end.
    DURASI_HILANG_MATI: int = 5 * 25
    # Veto: a gap during which the players move this fast (median of the per-frame mean, m/s) is
    # taken to be live play the tracker lost the ball in. Starting point, not tuned.
    V_GAP_HIDUP: float = 2.0

    # Score rule, for stoppages where the ball stays visible. Player speed alone barely separates
    # stoppages from play on this data (mean speed < 1 m/s on 24% of all frames and 26% of
    # set-piece frames), so it no longer counts; it is still reported by pipeline/diagnose.py.
    BOBOT_LUAR: float = 2.0         # ball seen beyond a line
    BOBOT_PEMAIN_DIAM: float = 0.0  # players slow
    BOBOT_BOLA_DIAM: float = 1.0    # ball at rest (also direct evidence, frame for frame)
    BOBOT_SET_PIECE: float = 1.0    # near the start of a SET PIECES tag (accurate to ~1 s)
    BOBOT_POLA: float = 1.0         # player layout matches a restart (Bagian 6)

    JENDELA_SET_PIECE: int = 25    # frames either side of a SET PIECES tag start that get its weight
    JENDELA_SKOR: int = 25         # half-width of the window the score is averaged over (+/- 1 s)
    # Starting point only -- Tahap 3 sets it on external data. With the default weights a ball
    # seen out of play (2) passes alone; a tag, a resting ball or a restart layout (1 each) need
    # a second piece of evidence.
    AMBANG_MATI: float = 1.5
    DURASI_MATI_MIN: int = 25      # dead stretches shorter than this are dropped

    # -- Bagian 6: restart triggers [ASLI] -------------------------------------------------------
    R_POJOK: float = 2.0       # player this close to a corner flag = corner
    R_TITIK: float = 1.5       # player this close to the centre / penalty spot
    TOL_SEPARUH: float = 1.0   # tolerance (m) for "every player in their own half" at kickoff
    JENDELA_RESTART: int = 12  # frames at the end of a dead stretch the layout is read over

    # -- Bagian 7: labels [ADAPTASI] -------------------------------------------------------------
    # A ball missing at least this long inside live play lets a Smart Tagging phase of the other
    # team switch possession ("kecuali ada bukti pergantian").
    GAP_HILANG_PANJANG: int = 3 * 25
    PAKAI_SMART_TAGGING: bool = True

    # -- Bagian 8: tidy spells [ADAPTASI] --------------------------------------------------------
    SPELL_MIN: int = 25   # H/A spells shorter than this are handed to a neighbour

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)


# Smart Tagging spells the clubs its own way ("Atletic Club"), so the tag is mapped to a side by
# keyword rather than by comparing with the ATD metadata names. Lower case, substring match.
TIM_SMART_TAGGING = {"barcelona": H, "atletic": A, "athletic": A}


def team_of_tag(value):
    """H, A or NONE for a Smart Tagging Team value. 'N/A' (every throw-in) is NONE."""
    text = (value or "").strip().lower()
    for keyword, team in TIM_SMART_TAGGING.items():
        if keyword in text:
            return team
    return NONE
