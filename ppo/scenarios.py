"""Fixed Austin PPO scenarios and deterministic role queues."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Sequence
import numpy as np
from utils import *

MAP_NAME = "Austin"
EGO_RACELINE = "raceline1"
OPPONENT_RACELINES = ("raceline0", "raceline1", "raceline2")
OPPONENT_SPEED_SCALES = (0.5, 0.6, 0.7, 0.8)
ORDINARY_INTERVAL_IDX = 15
SIM_DURATION = 8.0
TIMESTEP = 0.01
H1_SCENARIO_COUNT = 482
ORDINARY_SCENARIO_COUNT = 600

ORDINARY_STARTPOINTS = (
    21, 63, 110, 151, 189, 231, 272, 319, 356, 398,
    440, 487, 519, 571, 613, 650, 692, 739, 780, 823,
    861, 904, 949, 989, 1032, 1064, 1106, 1149, 1189, 1234,
    1272, 1315, 1356, 1404, 1441, 1488, 1525, 1567, 1608, 1656,
    1703, 1745, 1787, 1824, 1865, 1912, 1954, 1997, 2033, 2075,
)

H1_SCENARIOS = (
    ('v12-sp000-ego0010-raceline0-i10-v080', 0, 10, 16, 'raceline0', 0.8, 10),
    ('v12-sp001-ego0031-raceline0-i08-v080', 1, 31, 33, 'raceline0', 0.8, 8),
    ('v12-sp001-ego0031-raceline2-i12-v080', 1, 31, 49, 'raceline2', 0.8, 12),
    ('v12-sp002-ego0052-raceline0-i08-v075', 2, 52, 54, 'raceline0', 0.75, 8),
    ('v12-sp002-ego0052-raceline2-i15-v080', 2, 52, 73, 'raceline2', 0.8, 15),
    ('v12-sp003-ego0073-raceline2-i12-v080', 3, 73, 91, 'raceline2', 0.8, 12),
    ('v12-sp003-ego0073-raceline2-i15-v065', 3, 73, 94, 'raceline2', 0.65, 15),
    ('v12-sp003-ego0073-raceline2-i15-v080', 3, 73, 94, 'raceline2', 0.8, 15),
    ('v12-sp004-ego0094-raceline0-i08-v085', 4, 94, 96, 'raceline0', 0.85, 8),
    ('v12-sp004-ego0094-raceline0-i12-v080', 4, 94, 100, 'raceline0', 0.8, 12),
    ('v12-sp004-ego0094-raceline0-i15-v080', 4, 94, 103, 'raceline0', 0.8, 15),
    ('v12-sp004-ego0094-raceline2-i10-v070', 4, 94, 110, 'raceline2', 0.7, 10),
    ('v12-sp004-ego0094-raceline2-i15-v060', 4, 94, 115, 'raceline2', 0.6, 15),
    ('v12-sp004-ego0094-raceline2-i15-v080', 4, 94, 115, 'raceline2', 0.8, 15),
    ('v12-sp005-ego0115-raceline0-i08-v085', 5, 115, 117, 'raceline0', 0.85, 8),
    ('v12-sp005-ego0115-raceline0-i15-v060', 5, 115, 124, 'raceline0', 0.6, 15),
    ('v12-sp005-ego0115-raceline0-i15-v080', 5, 115, 124, 'raceline0', 0.8, 15),
    ('v12-sp006-ego0136-raceline0-i08-v085', 6, 136, 138, 'raceline0', 0.85, 8),
    ('v12-sp006-ego0136-raceline0-i15-v065', 6, 136, 145, 'raceline0', 0.65, 15),
    ('v12-sp006-ego0136-raceline0-i15-v080', 6, 136, 145, 'raceline0', 0.8, 15),
    ('v12-sp006-ego0136-raceline1-i08-v050', 6, 136, 144, 'raceline1', 0.5, 8),
    ('v12-sp006-ego0136-raceline1-i10-v045', 6, 136, 146, 'raceline1', 0.45, 10),
    ('v12-sp006-ego0136-raceline2-i08-v050', 6, 136, 150, 'raceline2', 0.5, 8),
    ('v12-sp006-ego0136-raceline2-i10-v075', 6, 136, 152, 'raceline2', 0.75, 10),
    ('v12-sp006-ego0136-raceline2-i12-v075', 6, 136, 154, 'raceline2', 0.75, 12),
    ('v12-sp006-ego0136-raceline2-i15-v075', 6, 136, 157, 'raceline2', 0.75, 15),
    ('v12-sp007-ego0157-raceline0-i08-v080', 7, 157, 158, 'raceline0', 0.8, 8),
    ('v12-sp007-ego0157-raceline0-i08-v085', 7, 157, 158, 'raceline0', 0.85, 8),
    ('v12-sp007-ego0157-raceline0-i12-v085', 7, 157, 162, 'raceline0', 0.85, 12),
    ('v12-sp007-ego0157-raceline0-i15-v080', 7, 157, 165, 'raceline0', 0.8, 15),
    ('v12-sp008-ego0178-raceline0-i08-v080', 8, 178, 179, 'raceline0', 0.8, 8),
    ('v12-sp008-ego0178-raceline0-i10-v075', 8, 178, 181, 'raceline0', 0.75, 10),
    ('v12-sp008-ego0178-raceline0-i12-v075', 8, 178, 183, 'raceline0', 0.75, 12),
    ('v12-sp008-ego0178-raceline0-i15-v080', 8, 178, 186, 'raceline0', 0.8, 15),
    ('v12-sp008-ego0178-raceline2-i15-v070', 8, 178, 201, 'raceline2', 0.7, 15),
    ('v12-sp009-ego0199-raceline0-i10-v080', 9, 199, 202, 'raceline0', 0.8, 10),
    ('v12-sp009-ego0199-raceline0-i12-v075', 9, 199, 204, 'raceline0', 0.75, 12),
    ('v12-sp009-ego0199-raceline2-i12-v045', 9, 199, 219, 'raceline2', 0.45, 12),
    ('v12-sp009-ego0199-raceline2-i15-v080', 9, 199, 222, 'raceline2', 0.8, 15),
    ('v12-sp010-ego0220-raceline0-i08-v085', 10, 220, 221, 'raceline0', 0.85, 8),
    ('v12-sp010-ego0220-raceline0-i10-v075', 10, 220, 223, 'raceline0', 0.75, 10),
    ('v12-sp010-ego0220-raceline1-i10-v045', 10, 220, 230, 'raceline1', 0.45, 10),
    ('v12-sp010-ego0220-raceline1-i15-v060', 10, 220, 235, 'raceline1', 0.6, 15),
    ('v12-sp010-ego0220-raceline2-i10-v055', 10, 220, 238, 'raceline2', 0.55, 10),
    ('v12-sp010-ego0220-raceline2-i10-v060', 10, 220, 238, 'raceline2', 0.6, 10),
    ('v12-sp010-ego0220-raceline2-i10-v065', 10, 220, 238, 'raceline2', 0.65, 10),
    ('v12-sp010-ego0220-raceline2-i10-v070', 10, 220, 238, 'raceline2', 0.7, 10),
    ('v12-sp010-ego0220-raceline2-i12-v045', 10, 220, 240, 'raceline2', 0.45, 12),
    ('v12-sp010-ego0220-raceline2-i12-v075', 10, 220, 240, 'raceline2', 0.75, 12),
    ('v12-sp011-ego0241-raceline0-i08-v055', 11, 241, 239, 'raceline0', 0.55, 8),
    ('v12-sp011-ego0241-raceline0-i08-v060', 11, 241, 239, 'raceline0', 0.6, 8),
    ('v12-sp011-ego0241-raceline0-i10-v055', 11, 241, 241, 'raceline0', 0.55, 10),
    ('v12-sp011-ego0241-raceline0-i10-v080', 11, 241, 241, 'raceline0', 0.8, 10),
    ('v12-sp011-ego0241-raceline2-i12-v070', 11, 241, 265, 'raceline2', 0.7, 12),
    ('v12-sp012-ego0262-raceline0-i08-v065', 12, 262, 261, 'raceline0', 0.65, 8),
    ('v12-sp012-ego0262-raceline0-i08-v085', 12, 262, 261, 'raceline0', 0.85, 8),
    ('v12-sp012-ego0262-raceline0-i10-v075', 12, 262, 263, 'raceline0', 0.75, 10),
    ('v12-sp012-ego0262-raceline0-i12-v070', 12, 262, 265, 'raceline0', 0.7, 12),
    ('v12-sp012-ego0262-raceline0-i15-v070', 12, 262, 268, 'raceline0', 0.7, 15),
    ('v12-sp013-ego0283-raceline0-i10-v080', 13, 283, 287, 'raceline0', 0.8, 10),
    ('v12-sp013-ego0283-raceline0-i12-v080', 13, 283, 289, 'raceline0', 0.8, 12),
    ('v12-sp013-ego0283-raceline2-i08-v075', 13, 283, 297, 'raceline2', 0.75, 8),
    ('v12-sp014-ego0305-raceline0-i08-v085', 14, 305, 307, 'raceline0', 0.85, 8),
    ('v12-sp014-ego0305-raceline0-i12-v085', 14, 305, 311, 'raceline0', 0.85, 12),
    ('v12-sp014-ego0305-raceline0-i15-v080', 14, 305, 314, 'raceline0', 0.8, 15),
    ('v12-sp014-ego0305-raceline0-i15-v085', 14, 305, 314, 'raceline0', 0.85, 15),
    ('v12-sp014-ego0305-raceline2-i10-v045', 14, 305, 321, 'raceline2', 0.45, 10),
    ('v12-sp015-ego0325-raceline0-i08-v085', 15, 325, 326, 'raceline0', 0.85, 8),
    ('v12-sp015-ego0325-raceline0-i10-v080', 15, 325, 328, 'raceline0', 0.8, 10),
    ('v12-sp015-ego0325-raceline0-i10-v085', 15, 325, 328, 'raceline0', 0.85, 10),
    ('v12-sp015-ego0325-raceline0-i12-v080', 15, 325, 330, 'raceline0', 0.8, 12),
    ('v12-sp015-ego0325-raceline0-i15-v080', 15, 325, 333, 'raceline0', 0.8, 15),
    ('v12-sp016-ego0347-raceline0-i10-v075', 16, 347, 348, 'raceline0', 0.75, 10),
    ('v12-sp016-ego0347-raceline0-i12-v075', 16, 347, 350, 'raceline0', 0.75, 12),
    ('v12-sp017-ego0367-raceline0-i15-v065', 17, 367, 373, 'raceline0', 0.65, 15),
    ('v12-sp018-ego0389-raceline0-i08-v055', 18, 389, 389, 'raceline0', 0.55, 8),
    ('v12-sp018-ego0389-raceline0-i08-v060', 18, 389, 389, 'raceline0', 0.6, 8),
    ('v12-sp018-ego0389-raceline0-i10-v055', 18, 389, 391, 'raceline0', 0.55, 10),
    ('v12-sp018-ego0389-raceline0-i10-v070', 18, 389, 391, 'raceline0', 0.7, 10),
    ('v12-sp018-ego0389-raceline0-i12-v065', 18, 389, 393, 'raceline0', 0.65, 12),
    ('v12-sp018-ego0389-raceline0-i12-v070', 18, 389, 393, 'raceline0', 0.7, 12),
    ('v12-sp018-ego0389-raceline0-i15-v065', 18, 389, 396, 'raceline0', 0.65, 15),
    ('v12-sp018-ego0389-raceline2-i08-v075', 18, 389, 406, 'raceline2', 0.75, 8),
    ('v12-sp018-ego0389-raceline2-i10-v075', 18, 389, 408, 'raceline2', 0.75, 10),
    ('v12-sp018-ego0389-raceline2-i12-v075', 18, 389, 410, 'raceline2', 0.75, 12),
    ('v12-sp019-ego0409-raceline0-i08-v050', 19, 409, 411, 'raceline0', 0.5, 8),
    ('v12-sp019-ego0409-raceline0-i08-v065', 19, 409, 411, 'raceline0', 0.65, 8),
    ('v12-sp019-ego0409-raceline0-i12-v045', 19, 409, 415, 'raceline0', 0.45, 12),
    ('v12-sp019-ego0409-raceline0-i12-v065', 19, 409, 415, 'raceline0', 0.65, 12),
    ('v12-sp019-ego0409-raceline0-i15-v065', 19, 409, 418, 'raceline0', 0.65, 15),
    ('v12-sp019-ego0409-raceline1-i10-v055', 19, 409, 419, 'raceline1', 0.55, 10),
    ('v12-sp019-ego0409-raceline2-i10-v075', 19, 409, 426, 'raceline2', 0.75, 10),
    ('v12-sp019-ego0409-raceline2-i12-v075', 19, 409, 428, 'raceline2', 0.75, 12),
    ('v12-sp020-ego0432-raceline0-i08-v065', 20, 432, 435, 'raceline0', 0.65, 8),
    ('v12-sp020-ego0432-raceline0-i10-v060', 20, 432, 437, 'raceline0', 0.6, 10),
    ('v12-sp020-ego0432-raceline0-i12-v060', 20, 432, 439, 'raceline0', 0.6, 12),
    ('v12-sp020-ego0432-raceline0-i15-v055', 20, 432, 442, 'raceline0', 0.55, 15),
    ('v12-sp020-ego0432-raceline2-i08-v075', 20, 432, 446, 'raceline2', 0.75, 8),
    ('v12-sp020-ego0432-raceline2-i15-v070', 20, 432, 453, 'raceline2', 0.7, 15),
    ('v12-sp021-ego0451-raceline0-i08-v070', 21, 451, 455, 'raceline0', 0.7, 8),
    ('v12-sp021-ego0451-raceline0-i08-v085', 21, 451, 455, 'raceline0', 0.85, 8),
    ('v12-sp021-ego0451-raceline0-i10-v070', 21, 451, 457, 'raceline0', 0.7, 10),
    ('v12-sp021-ego0451-raceline0-i10-v085', 21, 451, 457, 'raceline0', 0.85, 10),
    ('v12-sp021-ego0451-raceline0-i12-v055', 21, 451, 459, 'raceline0', 0.55, 12),
    ('v12-sp021-ego0451-raceline0-i12-v065', 21, 451, 459, 'raceline0', 0.65, 12),
    ('v12-sp021-ego0451-raceline0-i12-v070', 21, 451, 459, 'raceline0', 0.7, 12),
    ('v12-sp021-ego0451-raceline0-i12-v085', 21, 451, 459, 'raceline0', 0.85, 12),
    ('v12-sp021-ego0451-raceline0-i15-v060', 21, 451, 462, 'raceline0', 0.6, 15),
    ('v12-sp021-ego0451-raceline0-i15-v065', 21, 451, 462, 'raceline0', 0.65, 15),
    ('v12-sp021-ego0451-raceline0-i15-v080', 21, 451, 462, 'raceline0', 0.8, 15),
    ('v12-sp021-ego0451-raceline0-i15-v085', 21, 451, 462, 'raceline0', 0.85, 15),
    ('v12-sp021-ego0451-raceline2-i10-v075', 21, 451, 465, 'raceline2', 0.75, 10),
    ('v12-sp022-ego0476-raceline0-i08-v070', 22, 476, 478, 'raceline0', 0.7, 8),
    ('v12-sp022-ego0476-raceline0-i08-v085', 22, 476, 478, 'raceline0', 0.85, 8),
    ('v12-sp022-ego0476-raceline0-i10-v065', 22, 476, 480, 'raceline0', 0.65, 10),
    ('v12-sp022-ego0476-raceline0-i10-v070', 22, 476, 480, 'raceline0', 0.7, 10),
    ('v12-sp022-ego0476-raceline0-i10-v075', 22, 476, 480, 'raceline0', 0.75, 10),
    ('v12-sp022-ego0476-raceline0-i12-v085', 22, 476, 482, 'raceline0', 0.85, 12),
    ('v12-sp022-ego0476-raceline1-i15-v045', 22, 476, 491, 'raceline1', 0.45, 15),
    ('v12-sp022-ego0476-raceline2-i10-v070', 22, 476, 492, 'raceline2', 0.7, 10),
    ('v12-sp022-ego0476-raceline2-i15-v065', 22, 476, 497, 'raceline2', 0.65, 15),
    ('v12-sp023-ego0493-raceline1-i08-v050', 23, 493, 501, 'raceline1', 0.5, 8),
    ('v12-sp023-ego0493-raceline1-i10-v045', 23, 493, 503, 'raceline1', 0.45, 10),
    ('v12-sp023-ego0493-raceline1-i10-v050', 23, 493, 503, 'raceline1', 0.5, 10),
    ('v12-sp023-ego0493-raceline1-i12-v060', 23, 493, 505, 'raceline1', 0.6, 12),
    ('v12-sp024-ego0518-raceline0-i08-v080', 24, 518, 521, 'raceline0', 0.8, 8),
    ('v12-sp024-ego0518-raceline0-i08-v085', 24, 518, 521, 'raceline0', 0.85, 8),
    ('v12-sp024-ego0518-raceline0-i10-v080', 24, 518, 523, 'raceline0', 0.8, 10),
    ('v12-sp024-ego0518-raceline0-i12-v080', 24, 518, 525, 'raceline0', 0.8, 12),
    ('v12-sp024-ego0518-raceline2-i08-v045', 24, 518, 531, 'raceline2', 0.45, 8),
    ('v12-sp024-ego0518-raceline2-i08-v050', 24, 518, 531, 'raceline2', 0.5, 8),
    ('v12-sp024-ego0518-raceline2-i08-v055', 24, 518, 531, 'raceline2', 0.55, 8),
    ('v12-sp024-ego0518-raceline2-i10-v045', 24, 518, 533, 'raceline2', 0.45, 10),
    ('v12-sp024-ego0518-raceline2-i10-v050', 24, 518, 533, 'raceline2', 0.5, 10),
    ('v12-sp024-ego0518-raceline2-i10-v070', 24, 518, 533, 'raceline2', 0.7, 10),
    ('v12-sp025-ego0534-raceline0-i08-v075', 25, 534, 537, 'raceline0', 0.75, 8),
    ('v12-sp025-ego0534-raceline0-i08-v085', 25, 534, 537, 'raceline0', 0.85, 8),
    ('v12-sp025-ego0534-raceline0-i10-v085', 25, 534, 539, 'raceline0', 0.85, 10),
    ('v12-sp025-ego0534-raceline0-i12-v085', 25, 534, 541, 'raceline0', 0.85, 12),
    ('v12-sp025-ego0534-raceline0-i15-v080', 25, 534, 544, 'raceline0', 0.8, 15),
    ('v12-sp025-ego0534-raceline2-i08-v045', 25, 534, 547, 'raceline2', 0.45, 8),
    ('v12-sp025-ego0534-raceline2-i08-v050', 25, 534, 547, 'raceline2', 0.5, 8),
    ('v12-sp025-ego0534-raceline2-i08-v055', 25, 534, 547, 'raceline2', 0.55, 8),
    ('v12-sp025-ego0534-raceline2-i08-v060', 25, 534, 547, 'raceline2', 0.6, 8),
    ('v12-sp025-ego0534-raceline2-i08-v065', 25, 534, 547, 'raceline2', 0.65, 8),
    ('v12-sp025-ego0534-raceline2-i10-v045', 25, 534, 549, 'raceline2', 0.45, 10),
    ('v12-sp025-ego0534-raceline2-i10-v050', 25, 534, 549, 'raceline2', 0.5, 10),
    ('v12-sp025-ego0534-raceline2-i10-v055', 25, 534, 549, 'raceline2', 0.55, 10),
    ('v12-sp025-ego0534-raceline2-i10-v060', 25, 534, 549, 'raceline2', 0.6, 10),
    ('v12-sp025-ego0534-raceline2-i10-v065', 25, 534, 549, 'raceline2', 0.65, 10),
    ('v12-sp025-ego0534-raceline2-i12-v045', 25, 534, 551, 'raceline2', 0.45, 12),
    ('v12-sp025-ego0534-raceline2-i12-v050', 25, 534, 551, 'raceline2', 0.5, 12),
    ('v12-sp025-ego0534-raceline2-i12-v055', 25, 534, 551, 'raceline2', 0.55, 12),
    ('v12-sp025-ego0534-raceline2-i12-v060', 25, 534, 551, 'raceline2', 0.6, 12),
    ('v12-sp025-ego0534-raceline2-i12-v065', 25, 534, 551, 'raceline2', 0.65, 12),
    ('v12-sp025-ego0534-raceline2-i15-v045', 25, 534, 554, 'raceline2', 0.45, 15),
    ('v12-sp025-ego0534-raceline2-i15-v050', 25, 534, 554, 'raceline2', 0.5, 15),
    ('v12-sp025-ego0534-raceline2-i15-v070', 25, 534, 554, 'raceline2', 0.7, 15),
    ('v12-sp025-ego0534-raceline2-i15-v085', 25, 534, 554, 'raceline2', 0.85, 15),
    ('v12-sp026-ego0551-raceline0-i08-v080', 26, 551, 553, 'raceline0', 0.8, 8),
    ('v12-sp026-ego0551-raceline0-i08-v085', 26, 551, 553, 'raceline0', 0.85, 8),
    ('v12-sp026-ego0551-raceline0-i12-v080', 26, 551, 557, 'raceline0', 0.8, 12),
    ('v12-sp026-ego0551-raceline0-i12-v085', 26, 551, 557, 'raceline0', 0.85, 12),
    ('v12-sp027-ego0576-raceline0-i10-v080', 27, 576, 580, 'raceline0', 0.8, 10),
    ('v12-sp027-ego0576-raceline0-i12-v085', 27, 576, 582, 'raceline0', 0.85, 12),
    ('v12-sp027-ego0576-raceline0-i15-v085', 27, 576, 585, 'raceline0', 0.85, 15),
    ('v12-sp027-ego0576-raceline1-i15-v045', 27, 576, 591, 'raceline1', 0.45, 15),
    ('v12-sp027-ego0576-raceline1-i15-v050', 27, 576, 591, 'raceline1', 0.5, 15),
    ('v12-sp027-ego0576-raceline1-i15-v055', 27, 576, 591, 'raceline1', 0.55, 15),
    ('v12-sp027-ego0576-raceline2-i12-v085', 27, 576, 595, 'raceline2', 0.85, 12),
    ('v12-sp028-ego0593-raceline0-i12-v075', 28, 593, 599, 'raceline0', 0.75, 12),
    ('v12-sp028-ego0593-raceline1-i15-v045', 28, 593, 608, 'raceline1', 0.45, 15),
    ('v12-sp028-ego0593-raceline1-i15-v050', 28, 593, 608, 'raceline1', 0.5, 15),
    ('v12-sp028-ego0593-raceline2-i10-v085', 28, 593, 610, 'raceline2', 0.85, 10),
    ('v12-sp029-ego0618-raceline0-i08-v055', 29, 618, 620, 'raceline0', 0.55, 8),
    ('v12-sp029-ego0618-raceline0-i08-v080', 29, 618, 620, 'raceline0', 0.8, 8),
    ('v12-sp029-ego0618-raceline0-i10-v050', 29, 618, 622, 'raceline0', 0.5, 10),
    ('v12-sp029-ego0618-raceline0-i12-v045', 29, 618, 624, 'raceline0', 0.45, 12),
    ('v12-sp029-ego0618-raceline1-i15-v055', 29, 618, 633, 'raceline1', 0.55, 15),
    ('v12-sp029-ego0618-raceline2-i12-v075', 29, 618, 637, 'raceline2', 0.75, 12),
    ('v12-sp030-ego0635-raceline0-i08-v045', 30, 635, 638, 'raceline0', 0.45, 8),
    ('v12-sp030-ego0635-raceline0-i08-v050', 30, 635, 638, 'raceline0', 0.5, 8),
    ('v12-sp030-ego0635-raceline0-i10-v045', 30, 635, 640, 'raceline0', 0.45, 10),
    ('v12-sp030-ego0635-raceline0-i10-v050', 30, 635, 640, 'raceline0', 0.5, 10),
    ('v12-sp030-ego0635-raceline0-i12-v045', 30, 635, 642, 'raceline0', 0.45, 12),
    ('v12-sp030-ego0635-raceline1-i08-v045', 30, 635, 643, 'raceline1', 0.45, 8),
    ('v12-sp030-ego0635-raceline1-i08-v050', 30, 635, 643, 'raceline1', 0.5, 8),
    ('v12-sp030-ego0635-raceline1-i08-v055', 30, 635, 643, 'raceline1', 0.55, 8),
    ('v12-sp030-ego0635-raceline1-i10-v050', 30, 635, 645, 'raceline1', 0.5, 10),
    ('v12-sp030-ego0635-raceline1-i10-v055', 30, 635, 645, 'raceline1', 0.55, 10),
    ('v12-sp030-ego0635-raceline1-i12-v050', 30, 635, 647, 'raceline1', 0.5, 12),
    ('v12-sp030-ego0635-raceline1-i12-v055', 30, 635, 647, 'raceline1', 0.55, 12),
    ('v12-sp030-ego0635-raceline1-i15-v045', 30, 635, 650, 'raceline1', 0.45, 15),
    ('v12-sp030-ego0635-raceline2-i08-v080', 30, 635, 649, 'raceline2', 0.8, 8),
    ('v12-sp030-ego0635-raceline2-i10-v075', 30, 635, 651, 'raceline2', 0.75, 10),
    ('v12-sp030-ego0635-raceline2-i10-v080', 30, 635, 651, 'raceline2', 0.8, 10),
    ('v12-sp030-ego0635-raceline2-i15-v070', 30, 635, 656, 'raceline2', 0.7, 15),
    ('v12-sp030-ego0635-raceline2-i15-v075', 30, 635, 656, 'raceline2', 0.75, 15),
    ('v12-sp031-ego0660-raceline0-i15-v080', 31, 660, 671, 'raceline0', 0.8, 15),
    ('v12-sp031-ego0660-raceline0-i15-v085', 31, 660, 671, 'raceline0', 0.85, 15),
    ('v12-sp031-ego0660-raceline1-i08-v045', 31, 660, 668, 'raceline1', 0.45, 8),
    ('v12-sp031-ego0660-raceline1-i08-v050', 31, 660, 668, 'raceline1', 0.5, 8),
    ('v12-sp031-ego0660-raceline2-i08-v075', 31, 660, 672, 'raceline2', 0.75, 8),
    ('v12-sp031-ego0660-raceline2-i08-v080', 31, 660, 672, 'raceline2', 0.8, 8),
    ('v12-sp031-ego0660-raceline2-i10-v075', 31, 660, 674, 'raceline2', 0.75, 10),
    ('v12-sp031-ego0660-raceline2-i12-v075', 31, 660, 676, 'raceline2', 0.75, 12),
    ('v12-sp031-ego0660-raceline2-i15-v070', 31, 660, 679, 'raceline2', 0.7, 15),
    ('v12-sp032-ego0679-raceline0-i08-v045', 32, 679, 684, 'raceline0', 0.45, 8),
    ('v12-sp032-ego0679-raceline0-i12-v080', 32, 679, 688, 'raceline0', 0.8, 12),
    ('v12-sp032-ego0679-raceline2-i08-v080', 32, 679, 690, 'raceline2', 0.8, 8),
    ('v12-sp032-ego0679-raceline2-i08-v085', 32, 679, 690, 'raceline2', 0.85, 8),
    ('v12-sp032-ego0679-raceline2-i10-v080', 32, 679, 692, 'raceline2', 0.8, 10),
    ('v12-sp033-ego0702-raceline0-i10-v075', 33, 702, 709, 'raceline0', 0.75, 10),
    ('v12-sp033-ego0702-raceline0-i10-v080', 33, 702, 709, 'raceline0', 0.8, 10),
    ('v12-sp033-ego0702-raceline0-i12-v070', 33, 702, 711, 'raceline0', 0.7, 12),
    ('v12-sp033-ego0702-raceline1-i12-v055', 33, 702, 714, 'raceline1', 0.55, 12),
    ('v12-sp033-ego0702-raceline1-i15-v045', 33, 702, 717, 'raceline1', 0.45, 15),
    ('v12-sp033-ego0702-raceline1-i15-v055', 33, 702, 717, 'raceline1', 0.55, 15),
    ('v12-sp033-ego0702-raceline1-i15-v060', 33, 702, 717, 'raceline1', 0.6, 15),
    ('v12-sp033-ego0702-raceline2-i12-v080', 33, 702, 717, 'raceline2', 0.8, 12),
    ('v12-sp033-ego0702-raceline2-i15-v070', 33, 702, 720, 'raceline2', 0.7, 15),
    ('v12-sp034-ego0722-raceline0-i08-v070', 34, 722, 727, 'raceline0', 0.7, 8),
    ('v12-sp034-ego0722-raceline0-i08-v085', 34, 722, 727, 'raceline0', 0.85, 8),
    ('v12-sp034-ego0722-raceline0-i15-v050', 34, 722, 734, 'raceline0', 0.5, 15),
    ('v12-sp034-ego0722-raceline1-i08-v045', 34, 722, 730, 'raceline1', 0.45, 8),
    ('v12-sp034-ego0722-raceline1-i10-v045', 34, 722, 732, 'raceline1', 0.45, 10),
    ('v12-sp034-ego0722-raceline1-i12-v045', 34, 722, 734, 'raceline1', 0.45, 12),
    ('v12-sp034-ego0722-raceline1-i15-v045', 34, 722, 737, 'raceline1', 0.45, 15),
    ('v12-sp034-ego0722-raceline2-i10-v075', 34, 722, 735, 'raceline2', 0.75, 10),
    ('v12-sp034-ego0722-raceline2-i10-v080', 34, 722, 735, 'raceline2', 0.8, 10),
    ('v12-sp034-ego0722-raceline2-i15-v070', 34, 722, 740, 'raceline2', 0.7, 15),
    ('v12-sp036-ego0763-raceline2-i12-v070', 36, 763, 787, 'raceline2', 0.7, 12),
    ('v12-sp036-ego0763-raceline2-i15-v075', 36, 763, 790, 'raceline2', 0.75, 15),
    ('v12-sp041-ego0870-raceline0-i08-v085', 41, 870, 867, 'raceline0', 0.85, 8),
    ('v12-sp041-ego0870-raceline0-i10-v070', 41, 870, 869, 'raceline0', 0.7, 10),
    ('v12-sp042-ego0891-raceline0-i10-v070', 42, 891, 890, 'raceline0', 0.7, 10),
    ('v12-sp043-ego0912-raceline0-i08-v075', 43, 912, 909, 'raceline0', 0.75, 8),
    ('v12-sp043-ego0912-raceline0-i08-v080', 43, 912, 909, 'raceline0', 0.8, 8),
    ('v12-sp043-ego0912-raceline0-i10-v070', 43, 912, 911, 'raceline0', 0.7, 10),
    ('v12-sp044-ego0933-raceline0-i10-v070', 44, 933, 932, 'raceline0', 0.7, 10),
    ('v12-sp044-ego0933-raceline0-i10-v080', 44, 933, 932, 'raceline0', 0.8, 10),
    ('v12-sp044-ego0933-raceline0-i15-v065', 44, 933, 937, 'raceline0', 0.65, 15),
    ('v12-sp044-ego0933-raceline2-i15-v075', 44, 933, 960, 'raceline2', 0.75, 15),
    ('v12-sp045-ego0954-raceline0-i08-v075', 45, 954, 951, 'raceline0', 0.75, 8),
    ('v12-sp045-ego0954-raceline0-i10-v080', 45, 954, 953, 'raceline0', 0.8, 10),
    ('v12-sp045-ego0954-raceline0-i15-v085', 45, 954, 958, 'raceline0', 0.85, 15),
    ('v12-sp046-ego0975-raceline0-i08-v085', 46, 975, 972, 'raceline0', 0.85, 8),
    ('v12-sp046-ego0975-raceline0-i10-v070', 46, 975, 974, 'raceline0', 0.7, 10),
    ('v12-sp047-ego0996-raceline0-i08-v060', 47, 996, 993, 'raceline0', 0.6, 8),
    ('v12-sp047-ego0996-raceline0-i08-v070', 47, 996, 993, 'raceline0', 0.7, 8),
    ('v12-sp047-ego0996-raceline0-i08-v080', 47, 996, 993, 'raceline0', 0.8, 8),
    ('v12-sp047-ego0996-raceline0-i12-v070', 47, 996, 997, 'raceline0', 0.7, 12),
    ('v12-sp047-ego0996-raceline0-i15-v080', 47, 996, 1000, 'raceline0', 0.8, 15),
    ('v12-sp047-ego0996-raceline1-i12-v045', 47, 996, 1008, 'raceline1', 0.45, 12),
    ('v12-sp047-ego0996-raceline1-i15-v045', 47, 996, 1011, 'raceline1', 0.45, 15),
    ('v12-sp048-ego1017-raceline0-i08-v065', 48, 1017, 1014, 'raceline0', 0.65, 8),
    ('v12-sp048-ego1017-raceline0-i08-v085', 48, 1017, 1014, 'raceline0', 0.85, 8),
    ('v12-sp048-ego1017-raceline0-i10-v070', 48, 1017, 1016, 'raceline0', 0.7, 10),
    ('v12-sp048-ego1017-raceline1-i15-v045', 48, 1017, 1032, 'raceline1', 0.45, 15),
    ('v12-sp048-ego1017-raceline2-i08-v045', 48, 1017, 1037, 'raceline2', 0.45, 8),
    ('v12-sp048-ego1017-raceline2-i08-v050', 48, 1017, 1037, 'raceline2', 0.5, 8),
    ('v12-sp048-ego1017-raceline2-i08-v055', 48, 1017, 1037, 'raceline2', 0.55, 8),
    ('v12-sp049-ego1038-raceline0-i08-v065', 49, 1038, 1035, 'raceline0', 0.65, 8),
    ('v12-sp049-ego1038-raceline0-i08-v085', 49, 1038, 1035, 'raceline0', 0.85, 8),
    ('v12-sp049-ego1038-raceline0-i10-v065', 49, 1038, 1037, 'raceline0', 0.65, 10),
    ('v12-sp049-ego1038-raceline0-i15-v070', 49, 1038, 1042, 'raceline0', 0.7, 15),
    ('v12-sp049-ego1038-raceline2-i15-v080', 49, 1038, 1065, 'raceline2', 0.8, 15),
    ('v12-sp050-ego1058-raceline0-i08-v075', 50, 1058, 1049, 'raceline0', 0.75, 8),
    ('v12-sp050-ego1058-raceline0-i08-v080', 50, 1058, 1049, 'raceline0', 0.8, 8),
    ('v12-sp050-ego1058-raceline0-i10-v070', 50, 1058, 1051, 'raceline0', 0.7, 10),
    ('v12-sp050-ego1058-raceline0-i12-v070', 50, 1058, 1053, 'raceline0', 0.7, 12),
    ('v12-sp050-ego1058-raceline0-i12-v080', 50, 1058, 1053, 'raceline0', 0.8, 12),
    ('v12-sp050-ego1058-raceline0-i15-v065', 50, 1058, 1056, 'raceline0', 0.65, 15),
    ('v12-sp050-ego1058-raceline1-i12-v045', 50, 1058, 1070, 'raceline1', 0.45, 12),
    ('v12-sp051-ego1079-raceline0-i08-v085', 51, 1079, 1070, 'raceline0', 0.85, 8),
    ('v12-sp051-ego1079-raceline2-i08-v080', 51, 1079, 1105, 'raceline2', 0.8, 8),
    ('v12-sp052-ego1100-raceline0-i08-v085', 52, 1100, 1091, 'raceline0', 0.85, 8),
    ('v12-sp053-ego1121-raceline2-i08-v045', 53, 1121, 1147, 'raceline2', 0.45, 8),
    ('v12-sp053-ego1121-raceline2-i08-v050', 53, 1121, 1147, 'raceline2', 0.5, 8),
    ('v12-sp054-ego1142-raceline2-i12-v060', 54, 1142, 1172, 'raceline2', 0.6, 12),
    ('v12-sp054-ego1142-raceline2-i15-v050', 54, 1142, 1175, 'raceline2', 0.5, 15),
    ('v12-sp055-ego1163-raceline0-i15-v045', 55, 1163, 1158, 'raceline0', 0.45, 15),
    ('v12-sp055-ego1163-raceline1-i08-v045', 55, 1163, 1171, 'raceline1', 0.45, 8),
    ('v12-sp055-ego1163-raceline1-i10-v045', 55, 1163, 1173, 'raceline1', 0.45, 10),
    ('v12-sp055-ego1163-raceline1-i12-v045', 55, 1163, 1175, 'raceline1', 0.45, 12),
    ('v12-sp056-ego1184-raceline0-i08-v085', 56, 1184, 1171, 'raceline0', 0.85, 8),
    ('v12-sp056-ego1184-raceline0-i15-v075', 56, 1184, 1178, 'raceline0', 0.75, 15),
    ('v12-sp058-ego1226-raceline0-i12-v070', 58, 1226, 1217, 'raceline0', 0.7, 12),
    ('v12-sp059-ego1247-raceline0-i08-v080', 59, 1247, 1234, 'raceline0', 0.8, 8),
    ('v12-sp059-ego1247-raceline0-i10-v085', 59, 1247, 1236, 'raceline0', 0.85, 10),
    ('v12-sp060-ego1268-raceline0-i10-v075', 60, 1268, 1259, 'raceline0', 0.75, 10),
    ('v12-sp062-ego1310-raceline0-i08-v085', 62, 1310, 1301, 'raceline0', 0.85, 8),
    ('v12-sp063-ego1332-raceline0-i08-v075', 63, 1332, 1324, 'raceline0', 0.75, 8),
    ('v12-sp063-ego1332-raceline0-i08-v080', 63, 1332, 1324, 'raceline0', 0.8, 8),
    ('v12-sp063-ego1332-raceline1-i08-v070', 63, 1332, 1340, 'raceline1', 0.7, 8),
    ('v12-sp063-ego1332-raceline1-i08-v075', 63, 1332, 1340, 'raceline1', 0.75, 8),
    ('v12-sp063-ego1332-raceline1-i12-v045', 63, 1332, 1344, 'raceline1', 0.45, 12),
    ('v12-sp063-ego1332-raceline1-i15-v045', 63, 1332, 1347, 'raceline1', 0.45, 15),
    ('v12-sp064-ego1352-raceline0-i10-v080', 64, 1352, 1347, 'raceline0', 0.8, 10),
    ('v12-sp064-ego1352-raceline0-i12-v075', 64, 1352, 1349, 'raceline0', 0.75, 12),
    ('v12-sp064-ego1352-raceline1-i10-v045', 64, 1352, 1362, 'raceline1', 0.45, 10),
    ('v12-sp064-ego1352-raceline1-i12-v045', 64, 1352, 1364, 'raceline1', 0.45, 12),
    ('v12-sp064-ego1352-raceline1-i12-v060', 64, 1352, 1364, 'raceline1', 0.6, 12),
    ('v12-sp064-ego1352-raceline1-i15-v045', 64, 1352, 1367, 'raceline1', 0.45, 15),
    ('v12-sp064-ego1352-raceline1-i15-v060', 64, 1352, 1367, 'raceline1', 0.6, 15),
    ('v12-sp065-ego1374-raceline0-i08-v085', 65, 1374, 1369, 'raceline0', 0.85, 8),
    ('v12-sp065-ego1374-raceline0-i10-v070', 65, 1374, 1371, 'raceline0', 0.7, 10),
    ('v12-sp065-ego1374-raceline0-i10-v085', 65, 1374, 1371, 'raceline0', 0.85, 10),
    ('v12-sp065-ego1374-raceline0-i15-v075', 65, 1374, 1376, 'raceline0', 0.75, 15),
    ('v12-sp065-ego1374-raceline0-i15-v080', 65, 1374, 1376, 'raceline0', 0.8, 15),
    ('v12-sp065-ego1374-raceline1-i15-v045', 65, 1374, 1389, 'raceline1', 0.45, 15),
    ('v12-sp065-ego1374-raceline1-i15-v050', 65, 1374, 1389, 'raceline1', 0.5, 15),
    ('v12-sp065-ego1374-raceline2-i08-v085', 65, 1374, 1396, 'raceline2', 0.85, 8),
    ('v12-sp065-ego1374-raceline2-i10-v080', 65, 1374, 1398, 'raceline2', 0.8, 10),
    ('v12-sp065-ego1374-raceline2-i12-v080', 65, 1374, 1400, 'raceline2', 0.8, 12),
    ('v12-sp065-ego1374-raceline2-i12-v085', 65, 1374, 1400, 'raceline2', 0.85, 12),
    ('v12-sp065-ego1374-raceline2-i15-v075', 65, 1374, 1403, 'raceline2', 0.75, 15),
    ('v12-sp065-ego1374-raceline2-i15-v080', 65, 1374, 1403, 'raceline2', 0.8, 15),
    ('v12-sp066-ego1394-raceline0-i08-v075', 66, 1394, 1390, 'raceline0', 0.75, 8),
    ('v12-sp066-ego1394-raceline0-i08-v080', 66, 1394, 1390, 'raceline0', 0.8, 8),
    ('v12-sp066-ego1394-raceline0-i10-v075', 66, 1394, 1392, 'raceline0', 0.75, 10),
    ('v12-sp066-ego1394-raceline1-i12-v045', 66, 1394, 1406, 'raceline1', 0.45, 12),
    ('v12-sp066-ego1394-raceline1-i15-v045', 66, 1394, 1409, 'raceline1', 0.45, 15),
    ('v12-sp067-ego1417-raceline0-i12-v080', 67, 1417, 1417, 'raceline0', 0.8, 12),
    ('v12-sp067-ego1417-raceline0-i15-v075', 67, 1417, 1420, 'raceline0', 0.75, 15),
    ('v12-sp067-ego1417-raceline1-i15-v045', 67, 1417, 1432, 'raceline1', 0.45, 15),
    ('v12-sp067-ego1417-raceline1-i15-v050', 67, 1417, 1432, 'raceline1', 0.5, 15),
    ('v12-sp067-ego1417-raceline2-i08-v045', 67, 1417, 1439, 'raceline2', 0.45, 8),
    ('v12-sp067-ego1417-raceline2-i08-v050', 67, 1417, 1439, 'raceline2', 0.5, 8),
    ('v12-sp067-ego1417-raceline2-i10-v045', 67, 1417, 1441, 'raceline2', 0.45, 10),
    ('v12-sp068-ego1436-raceline1-i08-v045', 68, 1436, 1444, 'raceline1', 0.45, 8),
    ('v12-sp068-ego1436-raceline1-i10-v045', 68, 1436, 1446, 'raceline1', 0.45, 10),
    ('v12-sp068-ego1436-raceline1-i10-v050', 68, 1436, 1446, 'raceline1', 0.5, 10),
    ('v12-sp068-ego1436-raceline1-i12-v045', 68, 1436, 1448, 'raceline1', 0.45, 12),
    ('v12-sp068-ego1436-raceline1-i12-v050', 68, 1436, 1448, 'raceline1', 0.5, 12),
    ('v12-sp068-ego1436-raceline1-i15-v045', 68, 1436, 1451, 'raceline1', 0.45, 15),
    ('v12-sp068-ego1436-raceline2-i10-v085', 68, 1436, 1459, 'raceline2', 0.85, 10),
    ('v12-sp068-ego1436-raceline2-i12-v085', 68, 1436, 1461, 'raceline2', 0.85, 12),
    ('v12-sp068-ego1436-raceline2-i15-v085', 68, 1436, 1464, 'raceline2', 0.85, 15),
    ('v12-sp069-ego1459-raceline0-i15-v065', 69, 1459, 1454, 'raceline0', 0.65, 15),
    ('v12-sp069-ego1459-raceline0-i15-v085', 69, 1459, 1454, 'raceline0', 0.85, 15),
    ('v12-sp069-ego1459-raceline1-i08-v050', 69, 1459, 1467, 'raceline1', 0.5, 8),
    ('v12-sp069-ego1459-raceline1-i08-v055', 69, 1459, 1467, 'raceline1', 0.55, 8),
    ('v12-sp069-ego1459-raceline1-i10-v050', 69, 1459, 1469, 'raceline1', 0.5, 10),
    ('v12-sp069-ego1459-raceline1-i12-v050', 69, 1459, 1471, 'raceline1', 0.5, 12),
    ('v12-sp069-ego1459-raceline1-i12-v055', 69, 1459, 1471, 'raceline1', 0.55, 12),
    ('v12-sp069-ego1459-raceline1-i12-v060', 69, 1459, 1471, 'raceline1', 0.6, 12),
    ('v12-sp070-ego1478-raceline1-i10-v050', 70, 1478, 1488, 'raceline1', 0.5, 10),
    ('v12-sp070-ego1478-raceline1-i12-v050', 70, 1478, 1490, 'raceline1', 0.5, 12),
    ('v12-sp070-ego1478-raceline1-i15-v045', 70, 1478, 1493, 'raceline1', 0.45, 15),
    ('v12-sp071-ego1503-raceline0-i10-v085', 71, 1503, 1489, 'raceline0', 0.85, 10),
    ('v12-sp071-ego1503-raceline0-i12-v085', 71, 1503, 1491, 'raceline0', 0.85, 12),
    ('v12-sp071-ego1503-raceline0-i15-v085', 71, 1503, 1494, 'raceline0', 0.85, 15),
    ('v12-sp071-ego1503-raceline1-i08-v045', 71, 1503, 1511, 'raceline1', 0.45, 8),
    ('v12-sp071-ego1503-raceline1-i10-v045', 71, 1503, 1513, 'raceline1', 0.45, 10),
    ('v12-sp071-ego1503-raceline1-i10-v050', 71, 1503, 1513, 'raceline1', 0.5, 10),
    ('v12-sp071-ego1503-raceline1-i10-v060', 71, 1503, 1513, 'raceline1', 0.6, 10),
    ('v12-sp071-ego1503-raceline1-i10-v065', 71, 1503, 1513, 'raceline1', 0.65, 10),
    ('v12-sp071-ego1503-raceline1-i10-v070', 71, 1503, 1513, 'raceline1', 0.7, 10),
    ('v12-sp071-ego1503-raceline1-i12-v045', 71, 1503, 1515, 'raceline1', 0.45, 12),
    ('v12-sp071-ego1503-raceline1-i12-v050', 71, 1503, 1515, 'raceline1', 0.5, 12),
    ('v12-sp071-ego1503-raceline1-i12-v055', 71, 1503, 1515, 'raceline1', 0.55, 12),
    ('v12-sp071-ego1503-raceline1-i12-v065', 71, 1503, 1515, 'raceline1', 0.65, 12),
    ('v12-sp071-ego1503-raceline1-i15-v045', 71, 1503, 1518, 'raceline1', 0.45, 15),
    ('v12-sp071-ego1503-raceline1-i15-v050', 71, 1503, 1518, 'raceline1', 0.5, 15),
    ('v12-sp071-ego1503-raceline1-i15-v055', 71, 1503, 1518, 'raceline1', 0.55, 15),
    ('v12-sp071-ego1503-raceline1-i15-v060', 71, 1503, 1518, 'raceline1', 0.6, 15),
    ('v12-sp071-ego1503-raceline1-i15-v065', 71, 1503, 1518, 'raceline1', 0.65, 15),
    ('v12-sp072-ego1520-raceline0-i08-v075', 72, 1520, 1505, 'raceline0', 0.75, 8),
    ('v12-sp072-ego1520-raceline0-i08-v080', 72, 1520, 1505, 'raceline0', 0.8, 8),
    ('v12-sp072-ego1520-raceline0-i08-v085', 72, 1520, 1505, 'raceline0', 0.85, 8),
    ('v12-sp072-ego1520-raceline0-i10-v075', 72, 1520, 1507, 'raceline0', 0.75, 10),
    ('v12-sp072-ego1520-raceline0-i10-v080', 72, 1520, 1507, 'raceline0', 0.8, 10),
    ('v12-sp072-ego1520-raceline0-i10-v085', 72, 1520, 1507, 'raceline0', 0.85, 10),
    ('v12-sp072-ego1520-raceline0-i12-v080', 72, 1520, 1509, 'raceline0', 0.8, 12),
    ('v12-sp072-ego1520-raceline0-i12-v085', 72, 1520, 1509, 'raceline0', 0.85, 12),
    ('v12-sp072-ego1520-raceline0-i15-v080', 72, 1520, 1512, 'raceline0', 0.8, 15),
    ('v12-sp072-ego1520-raceline0-i15-v085', 72, 1520, 1512, 'raceline0', 0.85, 15),
    ('v12-sp072-ego1520-raceline1-i08-v050', 72, 1520, 1528, 'raceline1', 0.5, 8),
    ('v12-sp072-ego1520-raceline1-i08-v055', 72, 1520, 1528, 'raceline1', 0.55, 8),
    ('v12-sp072-ego1520-raceline1-i08-v060', 72, 1520, 1528, 'raceline1', 0.6, 8),
    ('v12-sp072-ego1520-raceline1-i10-v050', 72, 1520, 1530, 'raceline1', 0.5, 10),
    ('v12-sp072-ego1520-raceline1-i10-v055', 72, 1520, 1530, 'raceline1', 0.55, 10),
    ('v12-sp072-ego1520-raceline1-i10-v060', 72, 1520, 1530, 'raceline1', 0.6, 10),
    ('v12-sp072-ego1520-raceline1-i10-v065', 72, 1520, 1530, 'raceline1', 0.65, 10),
    ('v12-sp072-ego1520-raceline1-i12-v045', 72, 1520, 1532, 'raceline1', 0.45, 12),
    ('v12-sp072-ego1520-raceline1-i12-v050', 72, 1520, 1532, 'raceline1', 0.5, 12),
    ('v12-sp072-ego1520-raceline1-i12-v055', 72, 1520, 1532, 'raceline1', 0.55, 12),
    ('v12-sp072-ego1520-raceline1-i12-v065', 72, 1520, 1532, 'raceline1', 0.65, 12),
    ('v12-sp072-ego1520-raceline1-i15-v045', 72, 1520, 1535, 'raceline1', 0.45, 15),
    ('v12-sp072-ego1520-raceline1-i15-v050', 72, 1520, 1535, 'raceline1', 0.5, 15),
    ('v12-sp072-ego1520-raceline1-i15-v055', 72, 1520, 1535, 'raceline1', 0.55, 15),
    ('v12-sp072-ego1520-raceline1-i15-v060', 72, 1520, 1535, 'raceline1', 0.6, 15),
    ('v12-sp072-ego1520-raceline1-i15-v070', 72, 1520, 1535, 'raceline1', 0.7, 15),
    ('v12-sp072-ego1520-raceline2-i15-v085', 72, 1520, 1557, 'raceline2', 0.85, 15),
    ('v12-sp073-ego1544-raceline0-i08-v050', 73, 1544, 1531, 'raceline0', 0.5, 8),
    ('v12-sp073-ego1544-raceline0-i08-v055', 73, 1544, 1531, 'raceline0', 0.55, 8),
    ('v12-sp073-ego1544-raceline0-i08-v075', 73, 1544, 1531, 'raceline0', 0.75, 8),
    ('v12-sp073-ego1544-raceline0-i10-v070', 73, 1544, 1533, 'raceline0', 0.7, 10),
    ('v12-sp073-ego1544-raceline0-i10-v075', 73, 1544, 1533, 'raceline0', 0.75, 10),
    ('v12-sp073-ego1544-raceline0-i10-v080', 73, 1544, 1533, 'raceline0', 0.8, 10),
    ('v12-sp073-ego1544-raceline0-i15-v065', 73, 1544, 1538, 'raceline0', 0.65, 15),
    ('v12-sp073-ego1544-raceline1-i08-v045', 73, 1544, 1552, 'raceline1', 0.45, 8),
    ('v12-sp073-ego1544-raceline1-i08-v050', 73, 1544, 1552, 'raceline1', 0.5, 8),
    ('v12-sp073-ego1544-raceline1-i08-v060', 73, 1544, 1552, 'raceline1', 0.6, 8),
    ('v12-sp073-ego1544-raceline1-i10-v045', 73, 1544, 1554, 'raceline1', 0.45, 10),
    ('v12-sp073-ego1544-raceline1-i10-v050', 73, 1544, 1554, 'raceline1', 0.5, 10),
    ('v12-sp073-ego1544-raceline1-i12-v045', 73, 1544, 1556, 'raceline1', 0.45, 12),
    ('v12-sp074-ego1562-raceline0-i08-v045', 74, 1562, 1554, 'raceline0', 0.45, 8),
    ('v12-sp074-ego1562-raceline0-i08-v065', 74, 1562, 1554, 'raceline0', 0.65, 8),
    ('v12-sp074-ego1562-raceline0-i12-v085', 74, 1562, 1558, 'raceline0', 0.85, 12),
    ('v12-sp074-ego1562-raceline1-i08-v045', 74, 1562, 1570, 'raceline1', 0.45, 8),
    ('v12-sp074-ego1562-raceline1-i08-v050', 74, 1562, 1570, 'raceline1', 0.5, 8),
    ('v12-sp074-ego1562-raceline1-i10-v045', 74, 1562, 1572, 'raceline1', 0.45, 10),
    ('v12-sp074-ego1562-raceline2-i08-v080', 74, 1562, 1585, 'raceline2', 0.8, 8),
    ('v12-sp074-ego1562-raceline2-i10-v080', 74, 1562, 1587, 'raceline2', 0.8, 10),
    ('v12-sp074-ego1562-raceline2-i12-v075', 74, 1562, 1589, 'raceline2', 0.75, 12),
    ('v12-sp075-ego1587-raceline0-i08-v060', 75, 1587, 1581, 'raceline0', 0.6, 8),
    ('v12-sp075-ego1587-raceline2-i10-v085', 75, 1587, 1611, 'raceline2', 0.85, 10),
    ('v12-sp076-ego1603-raceline0-i08-v060', 76, 1603, 1597, 'raceline0', 0.6, 8),
    ('v12-sp076-ego1603-raceline0-i08-v075', 76, 1603, 1597, 'raceline0', 0.75, 8),
    ('v12-sp076-ego1603-raceline0-i08-v080', 76, 1603, 1597, 'raceline0', 0.8, 8),
    ('v12-sp076-ego1603-raceline0-i10-v060', 76, 1603, 1599, 'raceline0', 0.6, 10),
    ('v12-sp077-ego1619-raceline0-i08-v065', 77, 1619, 1613, 'raceline0', 0.65, 8),
    ('v12-sp077-ego1619-raceline1-i12-v045', 77, 1619, 1631, 'raceline1', 0.45, 12),
    ('v12-sp077-ego1619-raceline1-i15-v045', 77, 1619, 1634, 'raceline1', 0.45, 15),
    ('v12-sp077-ego1619-raceline2-i08-v050', 77, 1619, 1640, 'raceline2', 0.5, 8),
    ('v12-sp077-ego1619-raceline2-i10-v045', 77, 1619, 1642, 'raceline2', 0.45, 10),
    ('v12-sp088-ego1855-raceline0-i10-v085', 88, 1855, 1845, 'raceline0', 0.85, 10),
    ('v12-sp088-ego1855-raceline2-i10-v050', 88, 1855, 1884, 'raceline2', 0.5, 10),
    ('v12-sp090-ego1897-raceline2-i12-v085', 90, 1897, 1928, 'raceline2', 0.85, 12),
    ('v12-sp091-ego1918-raceline2-i08-v050', 91, 1918, 1945, 'raceline2', 0.5, 8),
    ('v12-sp091-ego1918-raceline2-i08-v055', 91, 1918, 1945, 'raceline2', 0.55, 8),
    ('v12-sp091-ego1918-raceline2-i08-v060', 91, 1918, 1945, 'raceline2', 0.6, 8),
    ('v12-sp091-ego1918-raceline2-i08-v085', 91, 1918, 1945, 'raceline2', 0.85, 8),
    ('v12-sp091-ego1918-raceline2-i10-v050', 91, 1918, 1947, 'raceline2', 0.5, 10),
    ('v12-sp091-ego1918-raceline2-i12-v085', 91, 1918, 1949, 'raceline2', 0.85, 12),
    ('v12-sp092-ego1939-raceline2-i12-v085', 92, 1939, 1970, 'raceline2', 0.85, 12),
    ('v12-sp092-ego1939-raceline2-i15-v080', 92, 1939, 1973, 'raceline2', 0.8, 15),
    ('v12-sp093-ego1960-raceline1-i10-v070', 93, 1960, 1970, 'raceline1', 0.7, 10),
    ('v12-sp093-ego1960-raceline1-i15-v080', 93, 1960, 1975, 'raceline1', 0.8, 15),
    ('v12-sp093-ego1960-raceline2-i10-v085', 93, 1960, 1989, 'raceline2', 0.85, 10),
    ('v12-sp094-ego1981-raceline0-i15-v085', 94, 1981, 1976, 'raceline0', 0.85, 15),
    ('v12-sp094-ego1981-raceline2-i10-v085', 94, 1981, 2010, 'raceline2', 0.85, 10),
    ('v12-sp095-ego2002-raceline0-i10-v070', 95, 2002, 1992, 'raceline0', 0.7, 10),
    ('v12-sp095-ego2002-raceline1-i12-v055', 95, 2002, 2014, 'raceline1', 0.55, 12),
    ('v12-sp095-ego2002-raceline2-i08-v050', 95, 2002, 2029, 'raceline2', 0.5, 8),
    ('v12-sp095-ego2002-raceline2-i08-v055', 95, 2002, 2029, 'raceline2', 0.55, 8),
    ('v12-sp095-ego2002-raceline2-i08-v060', 95, 2002, 2029, 'raceline2', 0.6, 8),
    ('v12-sp095-ego2002-raceline2-i10-v045', 95, 2002, 2031, 'raceline2', 0.45, 10),
    ('v12-sp095-ego2002-raceline2-i10-v050', 95, 2002, 2031, 'raceline2', 0.5, 10),
    ('v12-sp095-ego2002-raceline2-i12-v085', 95, 2002, 2033, 'raceline2', 0.85, 12),
    ('v12-sp095-ego2002-raceline2-i15-v080', 95, 2002, 2036, 'raceline2', 0.8, 15),
    ('v12-sp096-ego2023-raceline0-i08-v075', 96, 2023, 2011, 'raceline0', 0.75, 8),
    ('v12-sp096-ego2023-raceline2-i08-v050', 96, 2023, 2050, 'raceline2', 0.5, 8),
    ('v12-sp096-ego2023-raceline2-i08-v055', 96, 2023, 2050, 'raceline2', 0.55, 8),
    ('v12-sp096-ego2023-raceline2-i08-v060', 96, 2023, 2050, 'raceline2', 0.6, 8),
    ('v12-sp096-ego2023-raceline2-i10-v085', 96, 2023, 2052, 'raceline2', 0.85, 10),
    ('v12-sp097-ego2044-raceline1-i12-v060', 97, 2044, 2056, 'raceline1', 0.6, 12),
    ('v12-sp097-ego2044-raceline2-i08-v050', 97, 2044, 2071, 'raceline2', 0.5, 8),
    ('v12-sp097-ego2044-raceline2-i08-v055', 97, 2044, 2071, 'raceline2', 0.55, 8),
    ('v12-sp097-ego2044-raceline2-i08-v060', 97, 2044, 2071, 'raceline2', 0.6, 8),
    ('v12-sp098-ego2065-raceline1-i15-v050', 98, 2065, 2080, 'raceline1', 0.5, 15),
    ('v12-sp098-ego2065-raceline1-i15-v075', 98, 2065, 2080, 'raceline1', 0.75, 15),
    ('v12-sp098-ego2065-raceline2-i08-v045', 98, 2065, 2092, 'raceline2', 0.45, 8),
    ('v12-sp098-ego2065-raceline2-i08-v050', 98, 2065, 2092, 'raceline2', 0.5, 8),
    ('v12-sp099-ego2086-raceline0-i08-v045', 99, 2086, 0, 'raceline0', 0.45, 8),
    ('v12-sp099-ego2086-raceline0-i08-v050', 99, 2086, 0, 'raceline0', 0.5, 8),
    ('v12-sp099-ego2086-raceline0-i08-v055', 99, 2086, 0, 'raceline0', 0.55, 8),
    ('v12-sp099-ego2086-raceline0-i08-v060', 99, 2086, 0, 'raceline0', 0.6, 8),
    ('v12-sp099-ego2086-raceline0-i08-v065', 99, 2086, 0, 'raceline0', 0.65, 8),
    ('v12-sp099-ego2086-raceline0-i08-v070', 99, 2086, 0, 'raceline0', 0.7, 8),
    ('v12-sp099-ego2086-raceline0-i08-v075', 99, 2086, 0, 'raceline0', 0.75, 8),
    ('v12-sp099-ego2086-raceline0-i08-v080', 99, 2086, 0, 'raceline0', 0.8, 8),
    ('v12-sp099-ego2086-raceline0-i15-v050', 99, 2086, 7, 'raceline0', 0.5, 15),
    ('v12-sp099-ego2086-raceline1-i10-v070', 99, 2086, 0, 'raceline1', 0.7, 10),
    ('v12-sp099-ego2086-raceline1-i12-v065', 99, 2086, 2, 'raceline1', 0.65, 12),
    ('v12-sp099-ego2086-raceline2-i08-v085', 99, 2086, 2113, 'raceline2', 0.85, 8),
    ('v12-sp099-ego2086-raceline2-i10-v060', 99, 2086, 2115, 'raceline2', 0.6, 10),
    ('v12-sp099-ego2086-raceline2-i10-v065', 99, 2086, 2115, 'raceline2', 0.65, 10),
    ('v12-sp099-ego2086-raceline2-i10-v075', 99, 2086, 2115, 'raceline2', 0.75, 10),
    ('v12-sp099-ego2086-raceline2-i15-v055', 99, 2086, 3, 'raceline2', 0.55, 15),
)


@dataclass
class EpisodeResetSpec:
    poses: np.ndarray
    initial_speed_feature: float
    scenario: dict[str, Any]


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    pool: str
    startpoint_ordinal: int
    ego_idx: int
    opp_idx: int
    opp_raceline: str
    opp_speedscale: float
    interval_idx: int
    map_name: str = MAP_NAME
    ego_raceline: str = EGO_RACELINE
    sim_duration: float = SIM_DURATION
    timestep: float = TIMESTEP
    integrator: str = "RK4"

    def to_reset_spec(self, env_role: str) -> EpisodeResetSpec:
        scenario = asdict(self)
        scenario["opponent_speed_scale"] = self.opp_speedscale
        scenario["sampler_branch"] = env_role
        scenario["env_role"] = env_role
        poses, initial_speeds = load_positions_and_speeds_from_params(scenario, self.map_name)
        return EpisodeResetSpec(np.asarray(poses, dtype=np.float64), float(initial_speeds[0] * 0.9), scenario)


def ordinary_scenarios() -> tuple[ScenarioSpec, ...]:
    ego_waypoints = load_raceline_waypoints(MAP_NAME, f"{EGO_RACELINE}.csv")
    opponent_waypoints = {
        raceline: load_raceline_waypoints(MAP_NAME, f"{raceline}.csv") for raceline in OPPONENT_RACELINES
    }
    scenarios = []
    for ordinal, ego_idx in enumerate(ORDINARY_STARTPOINTS):
        ego_waypoint = ego_waypoints[ego_idx % len(ego_waypoints)]
        for opp_raceline in OPPONENT_RACELINES:
            if opp_raceline == EGO_RACELINE:
                mapped_index = ego_idx % len(ego_waypoints)
            else:
                mapped_index = int(find_corresponding_waypoint(ego_waypoint, opponent_waypoints[opp_raceline]))
            opp_idx = (mapped_index + ORDINARY_INTERVAL_IDX) % len(opponent_waypoints[opp_raceline])
            for speed_scale in OPPONENT_SPEED_SCALES:
                scenario_id = f"training-sp{ordinal:02d}-ego{ego_idx:04d}-{opp_raceline}-v{int(100 * speed_scale):03d}"
                scenarios.append(
                    ScenarioSpec(
                        scenario_id, "training", ordinal, ego_idx, int(opp_idx), opp_raceline,
                        speed_scale, ORDINARY_INTERVAL_IDX
                    )
                )
    if len(scenarios) != ORDINARY_SCENARIO_COUNT or len({row.scenario_id for row in scenarios}) != ORDINARY_SCENARIO_COUNT:
        raise RuntimeError("Ordinary scenario panel must contain 600 unique scenarios")
    return tuple(scenarios)


def h1_scenarios() -> tuple[ScenarioSpec, ...]:
    scenarios = tuple(
        ScenarioSpec(scenario_id, "train_austin_expanded_v1_2", ordinal, ego_idx, opp_idx, raceline, speed, interval)
        for scenario_id, ordinal, ego_idx, opp_idx, raceline, speed, interval in H1_SCENARIOS
    )
    if len(scenarios) != H1_SCENARIO_COUNT or len({row.scenario_id for row in scenarios}) != H1_SCENARIO_COUNT:
        raise RuntimeError("H1 scenario panel must contain 482 unique scenarios")
    return scenarios


class RoleScenarioQueue:

    def __init__(self, scenarios: Sequence[ScenarioSpec], seed_sequence: np.random.SeedSequence):
        self.scenarios = tuple(scenarios)
        self.rng = np.random.default_rng(seed_sequence)
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0
        self.cycle = 0
        self._start_cycle()

    def _start_cycle(self) -> None:
        self.order = np.asarray(self.rng.permutation(len(self.scenarios)), dtype=np.int64)
        self.cursor = 0
        self.cycle += 1

    def next(self) -> ScenarioSpec:
        if self.cursor == len(self.order):
            self._start_cycle()
        scenario = self.scenarios[int(self.order[self.cursor])]
        self.cursor += 1
        return scenario

    def state_dict(self) -> dict[str, Any]:
        return {
            "order": self.order.copy(),
            "cursor": self.cursor,
            "cycle": self.cycle,
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        order = np.asarray(state["order"], dtype=np.int64)
        if sorted(order.tolist()) != list(range(len(self.scenarios))):
            raise ValueError("Scenario queue order is invalid")
        cursor = int(state["cursor"])
        cycle = int(state["cycle"])
        if not 0 <= cursor <= len(order) or cycle <= 0:
            raise ValueError("Scenario queue cursor or cycle is invalid")
        self.order = order.copy()
        self.cursor = cursor
        self.cycle = cycle
        self.rng.bit_generator.state = deepcopy(state["rng_state"])


class ScenarioScheduler:

    def __init__(self, seed: int):
        hard_seed, ordinary_seed = np.random.SeedSequence(seed).spawn(2)
        self.hard = RoleScenarioQueue(h1_scenarios(), hard_seed)
        self.ordinary = RoleScenarioQueue(ordinary_scenarios(), ordinary_seed)

    def next(self, rank: int) -> EpisodeResetSpec:
        env_role = "hard" if rank % 2 == 0 else "ordinary"
        queue = self.hard if env_role == "hard" else self.ordinary
        return queue.next().to_reset_spec(env_role)

    def state_dict(self) -> dict[str, Any]:
        return {"hard": self.hard.state_dict(), "ordinary": self.ordinary.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.hard.load_state_dict(state["hard"])
        self.ordinary.load_state_dict(state["ordinary"])
