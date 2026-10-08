# -*- coding: utf-8 -*-
"""
Checks for the H2 part of tools/mps_columns.py -- centroid precision by
bootstrap over emission events, suspect marks on clusters, tau_0 and its
components, and the pre-specified parameters file -- against synthetic
axons of KNOWN truth (the research plan 03_plan.md, S3.6,
milestone H2; DECISIONES.md D-03, D-12).

The synthetic axon is the one of ``validate_rings_h1.py`` (imported as
``h1``: its module-level code builds the SMALL axon and sets
``QT_QPA_PLATFORM``, both cheap and harmless), so every H1 truth -- ring,
cluster and fluorophore of each localization, its frame number -- is
available here. Where a tolerance appears, the argument for it is next
to the check.

Points where this harness departs from the letter of the H2
specification, each with the number that forced it (measured on the
SMALL axon with the H1 code before H2 existed):

* "Every other cluster is clean" (marks, check c): Picasso's frame rule
  marks a cluster whose mean frame lies outside the middle 20-80 % of
  the movie. A SMALL cluster is 4 fluorophores with uniformly random
  start frames, so its mean frame has sd sqrt(1/48) = 0.14 of the movie
  and 3.8 % of the clusters fail the rule by chance: 2 of 72 do (true
  clusters 20 and 49, mean frame 0.82 and 0.90 of the movie). The harness
  therefore expects "burst" wherever the rule, re-implemented here from
  its definition, says so, and lists the chance cases in the [detail].
* The "edge" mark on the convex-hull path: ``build_rings`` forwards no
  ROI, ``analyze_axon`` then uses the convex hull of the ring's own
  localizations as the boundary, and on a ring of clusters the hull runs
  through the clusters: the runaway guard fires on EVERY ring of SMALL
  (24/24 clusters within eps of the hull, measured), and
  ``identify_bad_clusters`` empties ``edge_touching`` when it disables
  the criterion. The specification's "in edge_touching although kept" is
  therefore never true as the report stands. What the mark is for is
  the bias of a centroid and an area cut by the ROI; a hull cuts nothing.
  Encoded here: on the hull path no cluster carries "edge"; with a ROI
  forwarded (``build_rings(..., roi=...)``) whose guard fires, the kept
  clusters with a localization within ``eps_nm`` of the ROI boundary
  (recomputed from the ROI, since the report's set is empty) carry it,
  the others do not; with a ROI whose guard stays silent the touching
  clusters are removed by H1's criterion and nothing carries "edge".
  The ROI lives in the laboratory frame, so those runs use
  ``correct_tilt=False`` (identity frame); what a lab-frame ROI means
  after a rotation is left to the implementation and not tested.
* The few-events implant is one fluorophore in 12 consecutive frames. By
  construction 12 consecutive frames fall in one twentieth of the movie
  unless they straddle a window edge, so the implant's frames are 44994
  to 45005 (window edge at 45000 with n_frames = 60000: 6/6, share 50 %)
  and it carries "few_events" alone, as the specification says. With
  n_frames=None the length is max(frame) + 1 = 59787, the edges move
  (every 2989.35 frames) and the same implant is also a burst; the
  expectation is recomputed from the rule for that length.
* "Before and after H2 for the same inputs": the H1 fields of the plain
  H1 call on SMALL, as the H1 code (commit 8f39e0a) produced them before
  the H2 implementation, are embedded in this file (``H1_FP_*``,
  generated once from that code's output) and compared exactly (floats
  to 1e-9 nm), so the check depends on no file outside the harness and
  the reference cannot be regenerated from post-H1 code. Warnings may
  gain entries (H2 adds some), so the pre-H2 warnings are required to be
  still present, not identical.
* The period ``tau_components`` reports is the ring spacing per
  ``Ring.index`` step (the H2 review found the median over the LISTED
  rings doubled by a missing segment: 286 nm for rings 0, 1, 3 against
  191 nm), and ``recompute_tau`` follows that definition; on SMALL
  (indices 0, 1, 2) the two definitions agree, and section 4 checks the
  gapped cases built from the SMALL result.
* ``write_columns_params`` requires the provenance keys dataset_role,
  date and code_commit (03_plan S3.1) and refuses out-of-range values;
  ``load_columns_params`` refuses a repeated key and out-of-range values
  but not a missing provenance (a minimal hand-written file loads with
  an empty one). Both are H2 review additions, checked in section 5.
* The one-ring case of ``tau_components`` is the SMALL result truncated
  to its first ring (``dataclasses.replace``), not another build.
* ``n_clusters_used`` is read as "clusters entering the medians" = every
  cluster with finite rho and sigma_m (and no mark when
  ``exclude_suspect``) over all rings; on three rings every ring has a
  neighbour, so this equals "clusters that entered a pair".
* ``build_rings`` measured 5.9 s per call on SMALL on this machine (the
  task planned 25 s); this harness makes 7 calls, all on SMALL-sized
  axons (plain H1 call, n_frames = 60000, marks axon x 3, two ROI runs
  without tilt correction), plus 300 x 2 calls of
  ``bootstrap_centroid_sigma`` with 500 draws.
* The ROI runs also assert that ``bad_report.edge_touching`` is still
  empty when the guard fired: the mark must be recomputed from the ROI,
  not obtained by changing what ``identify_bad_clusters`` reports
  (``removed_edge_touching`` of the export reads that set).

The H1 harness itself (70 checks, ~10 min) is not re-run here by default:
``--with-h1`` runs it in a subprocess and requires "70 passed, 0 failed".

Run:  venv\\Scripts\\python.exe validate_columns_h2.py            (from the repo root)
      venv\\Scripts\\python.exe validate_columns_h2.py --with-h1

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import inspect
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import validate_rings_h1 as h1  # noqa: E402  builds SMALL, imports tools.mps_columns
from tools.cluster_quality import (  # noqa: E402
    PolygonROI,
    distance_to_roi_boundary,
    points_in_roi,
)
from tools.mps_columns import (  # noqa: E402  H1 names; the H2 ones go through require()
    Cluster,
    RingsParams,
    RingsResult,
    build_rings,
)

# ============================================================ H1 reference
# The H1 fields of ``validate_rings_h1.run_build_rings(SMALL)`` as the H1
# code produced them BEFORE H2 existed (commit 8f39e0a; 3 rings, 72
# clusters, 3 warnings), embedded here as every other validate_*.py
# embeds its truth, so that the "H1 unchanged" check depends on no file
# outside this harness and cannot be regenerated from post-H1 code. Floats
# are exact reprs. Rebuilt into the ``h1_fingerprint`` layout by
# ``h1_reference_fingerprint``.
H1_FP_CALL = 'validate_rings_h1.run_build_rings(SMALL): RingsParams() defaults, frame, lp, lpz'
H1_FP_SCALARS: Dict[str, Any] = {
    'n_events': 289,
    'link_radius_nm': 40.0,
    'link_lost_step_fraction': 0.0019304541362277093,
    'axis_start': 'lab-frame rings',
    'axis_candidate_beta_deg': [2.03515608019459, 2.0351203871921757],
    'axis_stages_beta_deg': [1.8573479316723203, 0.16509634951113256, 0.013726498079957159],
}
H1_FP_FRAME: Dict[str, Any] = {
    'method': 'plane_per_ring',
    'beta_deg': 2.03515608019459,
    'delta_beta_deg': 0.2362818350609867,
    'rotation': [
        [0.9994737464612171, -0.00023615471791943267, -0.03243723733193044],
        [-0.00023290743760225673, 0.9998954784657704, -0.014456067855181403],
        [0.032437260810744904, 0.014456015172145604, 0.9993692249296244],
    ],
    'origin_nm': [-0.6725145456663811, 1.9115652191195407, 1.6425427849430736],
    'n_locs_used': 1424,
}
H1_FP_WARNINGS: List[str] = [
    'Main-peak estimates disagree by 191 nm (density 192 nm vs weight 1 nm): the axial distribution has more than one comparable peak, so the 180 nm slab may not isolate the intended MPS segment.',
    'Ambiguous main peak: the two strongest axial components have nearly equal weights (0.34 vs 0.33). The 180 nm slab was centred on z = 192 nm, but a different MPS segment is almost equally supported by the data.',
    'Valley boundaries sit up to 4 nm away from the midpoints between means (median 3 nm).',
]
# (index, component_index, z_lo_nm, z_hi_nm, centre_z_nm, sigma_z_nm, lpz_median_nm,
#  length_nm, n_locs, n_events, loc_index_sha1, centroid_nm)
H1_FP_RINGS: List[Tuple[Any, ...]] = [
    (0, 0, -279.176554122761, -99.17655412276102, -189.17655412276102, 34.296901661821636,
     35.0, 4266.764660300921, 476, 96, 'df73499a4c38bab08be052908a0e774d66777252', [-0.5823415710097077, -0.5100562623404521]),
    (1, 1, -88.56181302212647, 91.43818697787353, 1.4381869778735292, 36.43324932072957,
     35.0, 4266.20717663637, 477, 98, 'bd224e4476537d98cba813d1267c92a4027782cd', [-1.58629807759419, -2.207410106351443]),
    (2, 2, 102.39239708778243, 282.39239708778246, 192.39239708778243, 32.74713694707117,
     35.0, 4274.09621578374, 476, 98, 'd1a0c59c9e27f9671e65b7ca8ffc5b8a68373ab0', [-1.3637983472190065, 2.5292013928535857]),
]
# (ring position, label, n_locs, n_events, centroid x, centroid y, area_nm2, rho_nm,
#  lpz_median_nm, loc_index_sha1)
H1_FP_CLUSTERS: List[Tuple[Any, ...]] = [
    (0, 0, 20, 4, 785.6908776048575, 95.40172324504437,
     828.8164819059016, 16.2425515243978, 35.0, 'f632a09fe28bfd451c948606a2f951fa773a9c21'),
    (0, 1, 20, 4, 727.9992863618069, 218.33550458469873,
     861.4583620846761, 16.559308958626502, 35.0, '7081d20fae752d4812bfdef5a238d49632f91d3b'),
    (0, 2, 20, 4, 619.8769093643529, 346.1693204961765,
     1082.8033976814772, 18.565210105878492, 35.0, 'e1db2e0feb93742f2e88fa66b8e45b976c7ef0f2'),
    (0, 3, 20, 4, 464.7878363029216, 444.39474553557193,
     641.869132617801, 14.293820012455058, 35.0, 'c103d508e3bdd3d769613eca143162906fe1b4d5'),
    (0, 4, 20, 4, 289.6107484380464, 524.2492768585029,
     649.6158432133411, 14.379817283833123, 35.0, '1cecef25ce004df342d6bc4551d235107031a8f4'),
    (0, 5, 20, 4, 69.43882236504191, 546.9627563004246,
     468.3758538562997, 12.210186924541103, 35.0, 'fc411db876ae0f4734dac8b64152d5c6526a537a'),
    (0, 6, 20, 4, -133.0536682664603, 546.8176810011167,
     1082.8758516108974, 18.565831225060734, 35.0, 'e6f96453f81748c584d90c0477f900271f926c1b'),
    (0, 7, 20, 4, -324.99775433773004, 501.1844097855508,
     698.802990031034, 14.914285105953656, 35.0, '8edff9c29599dd8eaa592650a286ca6b464d86ee'),
    (0, 8, 20, 4, -508.40426224504074, 424.66046850348255,
     1002.8771491818244, 17.866888683046945, 35.0, 'e97d7eb5bf50ee40c3eafc6921ac0b64f5ecd7b8'),
    (0, 9, 19, 4, -650.9822624411826, 319.7081260043601,
     899.7091111411853, 16.92295201097817, 35.0, '00a0b2c1cee6303ebe3255dd1d170467976426a7'),
    (0, 10, 20, 4, -747.5923856608354, 187.95038345453776,
     798.0307211887765, 15.938038399777053, 35.0, '4288842d02a4b79c93d6b758022aaa213bb7674a'),
    (0, 11, 19, 4, -791.615147543979, 54.75764651121916,
     776.0922203419051, 15.717437015148406, 35.0, '7711ae4b45d8c73f521757b8dff00373db742a45'),
    (0, 12, 19, 4, -795.3190450888368, -95.60671745720614,
     546.9432977542908, 13.194599609581015, 35.0, '0020987b0e9237d01309f13b57320b966977cb2f'),
    (0, 13, 20, 4, -722.0302295093596, -223.78118095227555,
     719.0401863567452, 15.128701196097014, 35.0, '5343bb648a4cd4f3a0db2951362f0b8177ed9452'),
    (0, 14, 20, 4, -618.0414682549572, -350.4865117742994,
     730.6584036216414, 15.250435839543565, 35.0, '0bbd7436fe5fd1b9ed0ce14a75dd4233d8c2163b'),
    (0, 15, 20, 4, -467.4919566011102, -453.01890030839775,
     887.4552883302283, 16.807313640840945, 35.0, '4a78f72750fe9f241e8ede838e43db5019d37138'),
    (0, 16, 20, 4, -274.22289062021974, -522.6204866642889,
     1497.839396201609, 21.835225845558178, 35.0, 'bb46553d5a2e529d2903156f2e6b986d378f5b4e'),
    (0, 17, 20, 4, -81.17538331462566, -546.1797065964511,
     839.7904950833538, 16.349728343560074, 35.0, '0b936779a74705e89481539bd4c541d76933321a'),
    (0, 18, 20, 4, 133.9185818784086, -543.0281584392264,
     654.0884956126471, 14.42923541261243, 35.0, '85a6ac742397f6a13e7260af5792fb040be82292'),
    (0, 19, 20, 4, 330.7187210111109, -498.5732819511378,
     923.8760345494621, 17.148728098764966, 35.0, '3b2fb92f56fb19aa563f49e7960ffc381a574100'),
    (0, 20, 20, 4, 513.2503698187634, -426.9388072936805,
     542.7272418757907, 13.143646624522647, 35.0, '7744817c37d227dbf1729f5a9ad8ebb73797afb5'),
    (0, 21, 19, 4, 661.8564109010556, -321.2994312290616,
     1135.8857063371258, 19.014827106812337, 35.0, '6d7e35d7dd3a533d91f3ba481e3d9dccda83c91d'),
    (0, 22, 20, 4, 746.8866067774879, -192.90094025382857,
     1113.8932322101687, 18.829849387759676, 35.0, 'c305ead04bf88a7e292c0c3515571f5a8be26154'),
    (0, 23, 20, 4, 785.6831571435458, -52.654201911661424,
     911.8405367724062, 17.0366621577641, 35.0, '679144d654b75cc6565ed3026b8a34f3dee63154'),
    (1, 0, 21, 5, -656.7891922401739, 320.4682266462726,
     929.9839827751335, 17.205321726428167, 35.0, '738a9aa9626386248682934011787ebe8aa36886'),
    (1, 1, 20, 4, 793.1306484480483, 90.09116333816915,
     613.6347633866675, 13.975908260003264, 35.0, '5fba5ef3d56006ed966b6f993fea59d5514e5d24'),
    (1, 2, 20, 4, 727.3591111374913, 224.7616845038845,
     798.4950075121842, 15.942674021616588, 35.0, 'acd27da6326c06f94d32434f44eaa50fe30304d5'),
    (1, 3, 20, 4, 619.2122668254303, 349.6166715120254,
     910.9614926739531, 17.028448227917366, 35.0, 'b34a9065467e5d05d2db4cb747bd91105902cf82'),
    (1, 4, 20, 4, 459.09448355130263, 443.16201598325307,
     499.043657454083, 12.60359194059185, 35.0, '645db0e61f2a594bfd51966cc8bede8ad91d883d'),
    (1, 5, 20, 4, 273.1590349936978, 505.8889645440986,
     707.7026855187819, 15.008956035629076, 35.0, '2b774f49aec4f67b5f55c3bf86a06c649d7798c8'),
    (1, 6, 20, 4, 80.6031614567969, 546.4652691513161,
     835.1022242458691, 16.30402692439917, 35.0, '2bb8b884084271ff9f522d4b1b12b7517da3a016'),
    (1, 7, 20, 4, -137.17813720304062, 533.5597994452147,
     946.9163617959199, 17.361245328283005, 35.0, '97e3d0d85d63a9c0b2a99442972f0a63dae5c442'),
    (1, 8, 20, 4, -331.4369133050205, 493.8304198065497,
     778.7940397759631, 15.74477189932269, 35.0, 'fac1ac6036ab9bfb001e37e2f8ac3395a9928eac'),
    (1, 9, 20, 4, -514.4523116522466, 419.84564394657417,
     763.7376614679844, 15.591832736921871, 35.0, '5e926f72d2a6e5301b5cfc1bd584c6d840dd161b'),
    (1, 10, 20, 4, -752.4349530349842, 197.00322063532656,
     1632.197728993877, 22.793522618179985, 35.0, '48be8bbb4d1c01e3645152a276db884ee7f03638'),
    (1, 11, 19, 4, -797.173081339496, 50.6319052523847,
     1043.5877048840962, 18.22592339401381, 35.0, 'e498068306f5abe96846220a895c06b2e7664a64'),
    (1, 12, 20, 4, -788.9466432517439, -99.187651654433,
     742.0509133296966, 15.368869241572513, 35.0, '4559bc908c865068216f329dbae2d699e8473201'),
    (1, 13, 20, 4, -728.5199155121929, -229.78009361682234,
     763.4190848041437, 15.588580499665152, 35.0, '212a7eb1471ac714d658714cd71f02262172c91c'),
    (1, 14, 21, 5, -617.388400178086, -359.62942589702516,
     753.2282372889723, 15.484185302490733, 35.0, 'ed804c750e82ae03f1f0bf6198af73dadbce979d'),
    (1, 15, 20, 4, -464.77699648045245, -443.63772856402136,
     467.55769454435324, 12.19951788124304, 35.0, '5068e047d61d921581dff34c21e8ac6d0c3bac06'),
    (1, 16, 19, 4, -273.29389981572484, -510.7883999917723,
     1095.022271888395, 18.66966562993325, 35.0, '009d16b5b76d3c177840a3d22020a581f3014336'),
    (1, 17, 20, 4, -73.78488950281393, -551.8427967805736,
     1436.7710070604194, 21.385472071702612, 35.0, '297f66cc770038cd97f1353a548f588812d21c73'),
    (1, 18, 20, 4, 143.95027340956193, -541.3003179319531,
     1147.0393632963278, 19.10795565148668, 35.0, '82ec456e94f6fbe4491d4759a9e687567a96659c'),
    (1, 19, 20, 4, 344.76137736141106, -498.27035203031494,
     1049.8400480299126, 18.2804394421903, 35.0, '29cfc34832f53f15865c0a3ab8876430fbcd3116'),
    (1, 20, 19, 4, 517.8821731291617, -416.0997095048036,
     883.5345140432415, 16.770145217157634, 35.0, 'e3a243b39bff9a4aab1fb6a9eb197712e4290c5a'),
    (1, 21, 19, 4, 652.72591028333, -323.5775751003097,
     895.0412627183786, 16.878995303799435, 35.0, '6773f37b15e6930fa9cecf8b9ce73f8263fbc67b'),
    (1, 22, 18, 4, 740.2376482803609, -192.3399109664718,
     1083.6384841590295, 18.572367715427028, 35.0, '6f4e2d4c509049e1f3c1d5d758b78bf55b1d2461'),
    (1, 23, 18, 4, 798.8782182174263, -49.0941300181923,
     1677.0501893485457, 23.104580820608717, 35.0, 'ca3350e9726dfc92345846fa8dcd3cb81ab0f77f'),
    (2, 0, 20, 5, -288.43925035324554, -511.7881917981198,
     898.0301094758919, 16.907154164343808, 35.0, 'a432cefc8e8ccb1d41086acd19a700381f4cf637'),
    (2, 1, 20, 4, 788.2322096358555, 88.88478538825404,
     889.613884895503, 16.8277418107305, 35.0, '564ccf352a2ad94965d7ac0525c625813ec16d7f'),
    (2, 2, 20, 4, 725.48984886516, 228.253249183463,
     1147.548025995783, 19.11219196076555, 35.0, 'bcb46ce506f2b9883daecba47ff048a34ceceb8e'),
    (2, 3, 20, 4, 618.7833671927484, 348.34756910733165,
     691.5135873585967, 14.836293717996409, 35.0, '241de288f0f4be56de41c5dc6b0d00aa4ad7ca9a'),
    (2, 4, 20, 4, 467.77488953851326, 452.7793369129075,
     1541.8389904021894, 22.153613555098673, 35.0, 'bbd6374b5e60164081ab7afe27c7a5b97ab97d5d'),
    (2, 5, 20, 4, 286.07758022960206, 520.1063235990061,
     811.9365089789032, 16.076299877818247, 35.0, 'a13d531092be5520d81f1f4a90098327deda0f9f'),
    (2, 6, 20, 4, 78.68632556391267, 549.4174492030132,
     1215.5072786769726, 19.66997670388138, 35.0, '003c1ae58c508c0818662621cad048c91ca66f51'),
    (2, 7, 20, 4, -124.66516795870398, 547.8673026414465,
     975.1967805930944, 17.61859177793126, 35.0, '4b7c641defab9996b47db3b49341b7d5d283bd0f'),
    (2, 8, 20, 4, -325.2226883206194, 498.1179422656054,
     837.0741079400027, 16.32326450272066, 35.0, '3a50be485eb1728fb82c556d3cd766af2894ddb4'),
    (2, 9, 20, 4, -497.81855245830957, 427.2601631255705,
     643.4971746436231, 14.311936012308138, 35.0, '513a88a6810e6b10f5f5945c204c28b9a23203e5'),
    (2, 10, 20, 4, -655.489336206879, 319.6165387398873,
     696.028312093896, 14.884646210216216, 35.0, 'c865b5c201f87278f3f1eb01623af2796ed3d227'),
    (2, 11, 20, 4, -754.2403616150182, 199.09658534009552,
     849.404708900152, 16.443050696691998, 35.0, '30d4cc10ad49cdd5710825aca1579c572ca1f279'),
    (2, 12, 19, 5, -810.5385774580624, 53.55192726917175,
     658.2079787345434, 14.474602129117589, 35.0, '0ca8ba0b91f94ed20bad455f8708a9321256e71d'),
    (2, 13, 20, 4, -787.3825848205754, -84.32809969914747,
     1151.066609816941, 19.141470203743292, 35.0, '36f07160e8b5d92dc3cb37fe678931805fb90029'),
    (2, 14, 20, 4, -737.3292760985989, -221.78969532680964,
     1438.9200752023864, 21.40145988864412, 35.0, 'e92feb88fb3e710f5902bd92eb5f76b063b40734'),
    (2, 15, 19, 4, -620.4939759157987, -349.35576754637276,
     1049.2998631285805, 18.275735826639803, 35.0, '98d1746f6b01e86e2067ccad895537f0d97892ee'),
    (2, 16, 20, 4, -461.3146454569313, -441.99399297510683,
     583.5370747002003, 13.628852476707014, 35.0, '092aa7d45ea2287f7d2c74a551daca4cec39b74f'),
    (2, 17, 19, 4, -77.2754355479657, -539.8063064408188,
     931.2649343061923, 17.21716687628831, 35.0, '4175605072884dd39dca7b4ba4753b83e749d033'),
    (2, 18, 20, 4, 123.36433692972844, -542.2403865279291,
     459.6064937829186, 12.095341695271445, 35.0, 'f9aa23d2c3d2cac66bd5235b00f825dc216f883c'),
    (2, 19, 20, 4, 320.1683058876256, -499.0494248872219,
     664.1637552306506, 14.539941173707252, 35.0, '1cc3bfba6875e77ec6bdc69ffc11c5ba7e987770'),
    (2, 20, 20, 4, 506.4499919434512, -425.08909207914473,
     874.2280431823333, 16.681589520307906, 35.0, '29c8ef2e65f8618969f8207e02d76cbfc3fd3626'),
    (2, 21, 20, 4, 651.0186549426181, -326.7759014853493,
     1705.6757328248916, 23.300932350057263, 35.0, '9ba00584d587ea8481007c3bf6c3d74ac9f6f9df'),
    (2, 22, 19, 4, 743.5971436057622, -192.27521163497056,
     594.4032225569671, 13.755159836199164, 35.0, 'fafb7add9bea81f44431b9da0b3f4cc152f4402a'),
    (2, 23, 20, 4, 796.5666588063732, -51.4820528282934,
     582.9688288890602, 13.622215002428, 35.0, '7b88f0a43d3d2630f9621cadd60046a773f5449e'),
]


def h1_reference_fingerprint() -> Dict[str, Any]:
    """The embedded pre-H2 reference in the layout of ``h1_fingerprint``."""
    rings: List[Dict[str, Any]] = []
    for t in H1_FP_RINGS:
        rings.append({
            "index": t[0], "component_index": t[1], "z_lo_nm": t[2], "z_hi_nm": t[3],
            "centre_z_nm": t[4], "sigma_z_nm": t[5], "lpz_median_nm": t[6], "length_nm": t[7],
            "n_locs": t[8], "n_events": t[9], "loc_index_sha1": t[10], "centroid_nm": t[11],
            "clusters": [],
        })
    for c in H1_FP_CLUSTERS:
        rings[c[0]]["clusters"].append({
            "label": c[1], "n_locs": c[2], "n_events": c[3], "centroid_nm": [c[4], c[5]],
            "area_nm2": c[6], "rho_nm": c[7], "lpz_median_nm": c[8], "loc_index_sha1": c[9],
        })
    return {"call": H1_FP_CALL, **H1_FP_SCALARS, "frame": dict(H1_FP_FRAME), "rings": rings,
            "warnings": list(H1_FP_WARNINGS)}


N_FRAMES = 60000                    # the acquisition length of make_synthetic_axon
MARKS = ("few_events", "burst", "edge")

PASSED = 0
FAILED = 0


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a harness; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def require(*names: str) -> Any:
    """The H2 names of ``tools.mps_columns``, or a RuntimeError naming what
    is missing, so that a check fails with the reason and never with a
    NameError further down (the H2 functions do not exist until the
    implementation lands; the harness is written first)."""
    import tools.mps_columns as mc
    missing = [n for n in names if not hasattr(mc, n)]
    if missing:
        raise RuntimeError(
            f"tools.mps_columns has no {', '.join(missing)} (H2 not implemented)")
    got = tuple(getattr(mc, n) for n in names)
    return got[0] if len(got) == 1 else got


def require_build_rings_kw(*kw: str) -> None:
    params = inspect.signature(build_rings).parameters
    missing = [k for k in kw if k not in params]
    if missing:
        raise RuntimeError(f"build_rings has no keyword {', '.join(missing)} (H2 not implemented)")


def need(state: Dict[str, Any], key: str) -> Any:
    """The result an earlier check of the section produced, or a clear error."""
    if key not in state:
        raise RuntimeError(f"'{key}' unavailable: the check producing it failed")
    return state[key]


def marks_of(cl: Any) -> Set[str]:
    return set(getattr(cl, "suspect", ()) or ())


def is_nan_float(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def all_clusters(res: Any) -> List[Tuple[Any, Any]]:
    return [(ring, cl) for ring in res.rings for cl in ring.clusters]


# ============================================================ first principles
def frame_rule_bursty(frames: NDArray[np.int64], total_frames: int) -> Tuple[bool, float, float]:
    """Picasso's frame-analysis rule, re-implemented from its definition:
    a cluster is bursty when its mean frame lies outside the middle
    20-80 % of the acquisition, or when one twentieth of the acquisition
    holds more than 80 % of its localizations."""
    fr = np.asarray(frames, dtype=np.int64)
    if fr.size == 0:
        return True, float("nan"), float("nan")
    mean_fraction = float(fr.mean()) / total_frames
    counts, _ = np.histogram(fr, bins=np.linspace(0, total_frames, 21))
    share = float(counts.max()) / fr.size
    return (not (0.2 <= mean_fraction <= 0.8 and share <= 0.8)), mean_fraction, share


def expected_marks(
    axon: h1.SyntheticAxon, res: Any, ring: Any, cl: Any, *,
    total_frames: Optional[int], roi: Optional[PolygonROI],
    frames_available: bool, min_events: int,
) -> Set[str]:
    """What the specification's three rules say about one kept cluster,
    computed from the truth arrays and the H1 fields, not from H2 code:
    few_events from the H1 ``n_events``; burst from the frame rule on
    the cluster's own frames; edge from the distance of its
    localizations to a forwarded ROI when the guard disabled the
    criterion (see the module docstring for the hull path)."""
    marks: Set[str] = set()
    li = np.asarray(cl.loc_index, dtype=np.intp)
    if frames_available:
        if cl.n_events is not None and cl.n_events < min_events:
            marks.add("few_events")
        if total_frames is not None and frame_rule_bursty(axon.frame[li], total_frames)[0]:
            marks.add("burst")
    if roi is not None and ring.analysis.bad_report.edge_criterion_disabled is not None:
        d = distance_to_roi_boundary(res.x_p[li], res.y_p[li], roi)
        if np.any(np.abs(d) <= ring.analysis.eps_nm):
            marks.add("edge")
    return marks


def majority_true_cluster(axon: h1.SyntheticAxon, cl: Any) -> int:
    ids, counts = np.unique(axon.cluster[np.asarray(cl.loc_index, dtype=np.intp)],
                            return_counts=True)
    return int(ids[int(np.argmax(counts))])


def find_cluster(axon: h1.SyntheticAxon, res: Any, true_id: int) -> Tuple[Any, Any]:
    hits = [(ring, cl) for ring, cl in all_clusters(res)
            if majority_true_cluster(axon, cl) == true_id]
    assert len(hits) == 1, f"true cluster {true_id} found {len(hits)} times among the kept clusters"
    return hits[0]


def bootstrap_sd_first_principles(
    units: NDArray[np.float64], n_draws: int, rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Bootstrap of the centroid over EQUAL-SIZE units (event means, or
    single localizations): the mean of the concatenated localizations of
    the drawn units is the mean of the unit means, so the draw is one
    ``mean`` over resampled rows. Returns (sigma, sx, sy) with ddof = 1
    over the draw means and sigma = sqrt((sx^2 + sy^2) / 2)."""
    u = np.asarray(units, dtype=np.float64).reshape(-1, 2)
    pick = rng.integers(0, u.shape[0], (n_draws, u.shape[0]))
    means = u[pick].mean(axis=1)
    sx, sy = means.std(axis=0, ddof=1)
    return float(math.sqrt((sx * sx + sy * sy) / 2.0)), float(sx), float(sy)


def reference_median_sigma(
    seed: int, *, n_clusters: int, n_events: int, locs_per_event: int,
    sigma_f_nm: float, lp_nm: float, n_draws: int, singleton: bool,
) -> float:
    """Median over ``n_clusters`` synthetic clusters of the first-principles
    bootstrap sigma; ``singleton`` treats every localization as its own
    unit (the naive bootstrap)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_clusters)
    for c in range(n_clusters):
        offsets = rng.normal(0.0, sigma_f_nm, (n_events, 1, 2))
        locs = offsets + rng.normal(0.0, lp_nm, (n_events, locs_per_event, 2))
        units = locs.reshape(-1, 2) if singleton else locs.mean(axis=1)
        out[c] = bootstrap_sd_first_principles(units, n_draws, rng)[0]
    return float(np.median(out))


# ============================================================ synthetic axons
BURST_ID = 24 + 3           # true cluster of the middle ring, parametric angle ~44 deg
FEW_ID = 24 + 15            # opposite side of the middle ring
FEW_FRAME0 = 44994          # 12 frames 44994..45005 straddle the window edge at 45000
BURST_FRAME0 = 30000        # 4 fluorophores x 5 frames within [30000, 30305): one twentieth


def subset_axon(axon: h1.SyntheticAxon, keep: NDArray[np.bool_]) -> h1.SyntheticAxon:
    """The axon restricted to the localizations ``keep`` (a ROI cut)."""
    arrays = ("x_nm", "y_nm", "z_nm", "true_x_nm", "true_y_nm", "true_z_nm", "frame",
              "ring", "cluster", "fluorophore", "lp_lateral_nm", "lpz_nm")
    return dataclasses.replace(axon, **{a: getattr(axon, a)[keep] for a in arrays})


def make_marks_axon(base: h1.SyntheticAxon, seed: int = 123) -> h1.SyntheticAxon:
    """
    SMALL with two implants in its middle ring.

    (a) True cluster ``BURST_ID`` keeps its 20 localizations and 4
        fluorophores but all its frames are moved into one twentieth of
        the acquisition (fluorophore k in frames 30000 + 100 k + 0..4:
        the 100-frame gaps keep the 4 events separate at max_dark_time
        1, so it is a burst with 4 events, not a few-events cluster).
    (b) True cluster ``FEW_ID`` becomes ONE fluorophore seen in 12
        consecutive frames (44994..45005): its first 12 rows are rewritten
        (one molecule at the cluster centre plus an 8 nm offset, 8 nm
        lateral and 35 nm axial noise, as in the generator) and its other
        8 rows are dropped.
    """
    rng = np.random.default_rng(seed)
    frame = base.frame.copy()
    idx_b = np.flatnonzero(base.cluster == BURST_ID)
    assert idx_b.size == 20 and np.all(np.diff(base.fluorophore[idx_b]) >= 0)
    assert np.unique(base.fluorophore[idx_b]).size == 4
    frame[idx_b] = BURST_FRAME0 + 100 * (np.arange(20) // 5) + np.arange(20) % 5

    idx_f = np.flatnonzero(base.cluster == FEW_ID)
    assert idx_f.size == 20
    kept_f = idx_f[:12]
    centre = np.array([base.true_x_nm[idx_f].mean(), base.true_y_nm[idx_f].mean(),
                       base.true_z_nm[idx_f].mean()])
    molecule = centre[:2] + rng.normal(0.0, 8.0, 2)
    x, y, z = base.x_nm.copy(), base.y_nm.copy(), base.z_nm.copy()
    tx, ty, tz = base.true_x_nm.copy(), base.true_y_nm.copy(), base.true_z_nm.copy()
    tx[kept_f], ty[kept_f], tz[kept_f] = molecule[0], molecule[1], centre[2]
    x[kept_f] = molecule[0] + rng.normal(0.0, base.lateral_noise_nm, 12)
    y[kept_f] = molecule[1] + rng.normal(0.0, base.lateral_noise_nm, 12)
    z[kept_f] = centre[2] + rng.normal(0.0, base.axial_noise_nm, 12)
    frame[kept_f] = FEW_FRAME0 + np.arange(12)
    fluor = base.fluorophore.copy()
    fluor[kept_f] = int(base.fluorophore.max()) + 1
    keep = np.ones(base.n_locs, dtype=bool)
    keep[idx_f[12:]] = False
    axon = dataclasses.replace(
        base, name="small-marks", x_nm=x, y_nm=y, z_nm=z,
        true_x_nm=tx, true_y_nm=ty, true_z_nm=tz, frame=frame, fluorophore=fluor)
    return subset_axon(axon, keep)


def ring_polygon_roi(axon: h1.SyntheticAxon, snug_until_deg: float,
                     snug_nm: float = 20.0, loose_nm: float = 200.0) -> PolygonROI:
    """A polygon ROI in the laboratory frame around the ring of clusters:
    the ellipse dilated by ``snug_nm`` for parametric angles below
    ``snug_until_deg`` (the clusters there sit on the boundary) and by
    ``loose_nm`` elsewhere (far from every cluster)."""
    a, b = axon.semi_axes_nm
    t = np.radians(np.arange(0.0, 360.0, 1.0))
    d = np.where(np.degrees(t) < snug_until_deg, snug_nm, loose_nm)
    rot = h1.rotation_z_to(axon.axis)
    lab = rot @ np.vstack([(a + d) * np.cos(t), (b + d) * np.sin(t), np.zeros_like(t)])
    return PolygonROI(vertices=np.column_stack([lab[0], lab[1]]))


SMALL = h1.SMALL
MARKS_AXON = make_marks_axon(SMALL)
ROI_SNUG = ring_polygon_roi(SMALL, 240.0)      # ~16 of 24 clusters touch: guard fires
ROI_MILD = ring_polygon_roi(SMALL, 90.0)       # ~6 of 24 touch: guard silent, they are removed
EDGE_AXON = dataclasses.replace(
    subset_axon(SMALL, points_in_roi(SMALL.x_nm, SMALL.y_nm, ROI_SNUG)), name="small-roi-snug")
MILD_AXON = dataclasses.replace(
    subset_axon(SMALL, points_in_roi(SMALL.x_nm, SMALL.y_nm, ROI_MILD)), name="small-roi-mild")


# ============================================================ H1 fingerprint
def h1_fingerprint(res: Any) -> Dict[str, Any]:
    """The H1 fields of a result, as plain JSON: what must not change."""
    def sha(a: Any) -> str:
        return hashlib.sha1(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()

    def fl(v: Any) -> Any:
        return None if v is None else float(v)

    rings = []
    for ring in res.rings:
        rings.append({
            "index": int(ring.index), "component_index": int(ring.component_index),
            "z_lo_nm": fl(ring.z_lo_nm), "z_hi_nm": fl(ring.z_hi_nm),
            "centre_z_nm": fl(ring.centre_z_nm), "sigma_z_nm": fl(ring.sigma_z_nm),
            "lpz_median_nm": fl(ring.lpz_median_nm), "length_nm": fl(ring.length_nm),
            "n_locs": int(ring.n_locs), "n_events": ring.n_events,
            "loc_index_sha1": sha(ring.loc_index),
            "centroid_nm": None if ring.centroid_nm is None else [float(v) for v in ring.centroid_nm],
            "clusters": [{
                "label": int(cl.label), "n_locs": int(cl.n_locs), "n_events": cl.n_events,
                "centroid_nm": [float(v) for v in cl.centroid_nm],
                "area_nm2": fl(cl.area_nm2), "rho_nm": fl(cl.rho_nm),
                "lpz_median_nm": fl(cl.lpz_median_nm),
                "loc_index_sha1": sha(cl.loc_index),
            } for cl in ring.clusters],
        })
    return {
        "call": "validate_rings_h1.run_build_rings(SMALL): RingsParams() defaults, frame, lp, lpz",
        "n_events": res.n_events, "link_radius_nm": fl(res.link_radius_nm),
        "link_lost_step_fraction": fl(res.link_lost_step_fraction),
        "axis_start": res.axis_start,
        "axis_candidate_beta_deg": [float(v) for v in res.axis_candidate_beta_deg],
        "axis_stages_beta_deg": [float(v) for v in res.axis_stages_beta_deg],
        "frame": {
            "method": res.frame.method, "beta_deg": fl(res.frame.beta_deg),
            "delta_beta_deg": fl(res.frame.delta_beta_deg),
            "rotation": np.asarray(res.frame.rotation, dtype=float).tolist(),
            "origin_nm": [float(v) for v in res.frame.origin_nm],
            "n_locs_used": int(res.frame.n_locs_used),
        },
        "rings": rings,
        "warnings": list(res.warnings),
    }


def diff_json(a: Any, b: Any, path: str, out: List[str], tol: float = 1e-9) -> None:
    if len(out) >= 8:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
            return
        for k in a:
            diff_json(a[k], b[k], f"{path}.{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (u, v) in enumerate(zip(a, b)):
            diff_json(u, v, f"{path}[{i}]", out, tol)
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            if a is not b:
                out.append(f"{path}: {a!r} vs {b!r}")
        elif math.isnan(float(a)) and math.isnan(float(b)):
            return
        elif not abs(float(a) - float(b)) <= tol:
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def load_fingerprint() -> Dict[str, Any]:
    """The pre-H2 reference (embedded; nothing is read from disk)."""
    return h1_reference_fingerprint()


# ============================================================ 0. truth
def test_truth() -> None:
    print("\n0. SYNTHETIC TRUTH (SMALL of validate_rings_h1, seed 0) and the H2 API")

    def api_present():
        require("bootstrap_centroid_sigma", "tau_components", "TauComponents",
                "ColumnsParams", "write_columns_params", "load_columns_params",
                "rings_params_from")
        require_build_rings_kw("n_frames")
        return "all H2 names importable; build_rings takes n_frames"

    def data_model():
        fields = {f.name: f for f in dataclasses.fields(Cluster)}
        for name in ("sigma_centroid_x_nm", "sigma_centroid_y_nm"):
            assert name in fields, f"Cluster.{name} missing"
            assert is_nan_float(fields[name].default), (name, fields[name].default)
        assert "n_bootstrap" in fields and fields["n_bootstrap"].default == 0
        assert fields["suspect"].default == ()
        p = RingsParams()
        assert p.n_bootstrap == 200, p.n_bootstrap
        assert p.min_events_per_cluster == 3, p.min_events_per_cluster
        rf = {f.name: f for f in dataclasses.fields(RingsResult)}
        assert "n_suspect" in rf, "RingsResult.n_suspect missing"
        assert rf["n_suspect"].default_factory is not dataclasses.MISSING
        assert rf["n_suspect"].default_factory() == {}
        sig = inspect.signature(build_rings).parameters
        assert "n_frames" in sig and sig["n_frames"].default is None
        tc = require("TauComponents")
        names = {f.name for f in dataclasses.fields(tc)}
        expected = {"tau0_nm", "rho_pair_median_nm", "sigma_m_median_nm", "sigma_pair_median_nm",
                    "period_nm", "delta_beta_deg", "lateral_shift_nm", "sigma_d_nm", "n_axons",
                    "n_clusters_used", "n_clusters_excluded", "n_pairs", "rho_quantiles_nm",
                    "sigma_m_quantiles_nm", "warnings"}
        assert expected <= names, sorted(expected - names)
        return ("Cluster: sigma_centroid_x/y_nm default NaN, n_bootstrap 0; RingsParams: "
                "n_bootstrap 200, min_events_per_cluster 3; RingsResult.n_suspect {}; "
                "TauComponents has every field of the spec")

    def implants():
        b = np.flatnonzero(MARKS_AXON.cluster == BURST_ID)
        f = np.flatnonzero(MARKS_AXON.cluster == FEW_ID)
        assert b.size == 20 and f.size == 12, (b.size, f.size)
        bursty, frac_b, share_b = frame_rule_bursty(MARKS_AXON.frame[b], N_FRAMES)
        assert bursty and share_b == 1.0 and 0.2 <= frac_b <= 0.8, (frac_b, share_b)
        few_b, frac_f, share_f = frame_rule_bursty(MARKS_AXON.frame[f], N_FRAMES)
        assert not few_b and share_f == 0.5, (frac_f, share_f)
        assert np.unique(MARKS_AXON.fluorophore[f]).size == 1
        assert np.array_equal(np.diff(MARKS_AXON.frame[f]), np.ones(11, dtype=np.int64))
        labels = DBSCAN(eps=25.0, min_samples=10).fit(
            np.column_stack([MARKS_AXON.x_nm[f], MARKS_AXON.y_nm[f]])).labels_
        assert np.count_nonzero(labels >= 0) >= 10, labels.tolist()
        assert MARKS_AXON.n_locs == SMALL.n_locs - 8
        return (f"burst implant: 20 locs, 4 fluorophores, mean frame {frac_b:.3f}, share "
                f"{share_b:.2f}; few-events implant: 12 locs, 1 fluorophore, frames "
                f"{FEW_FRAME0}..{FEW_FRAME0 + 11}, share {share_f:.2f}; DBSCAN keeps "
                f"{int(np.count_nonzero(labels >= 0))} of 12")

    def chance_bursts():
        # 4 uniform start frames: sd of the mean frame 0.144 of the movie,
        # P(outside 0.2-0.8) = 2 * (1 - Phi(2.08)) = 3.8 % per cluster,
        # 2.7 expected over 72; a bound of 8 is > 3 sd above that.
        hits = []
        for c in np.unique(SMALL.cluster):
            bursty, frac, share = frame_rule_bursty(SMALL.frame[SMALL.cluster == c], N_FRAMES)
            if bursty:
                hits.append((int(c), round(frac, 3), round(share, 2)))
        assert len(hits) <= 8, hits
        assert not any(c in (BURST_ID, FEW_ID) for c, _, _ in hits), hits
        return (f"{len(hits)} of {np.unique(SMALL.cluster).size} true clusters fail the frame "
                f"rule by chance (id, mean fraction, share): {hits}; none is an implant")

    def roi_cuts():
        n_snug = int(points_in_roi(SMALL.x_nm, SMALL.y_nm, ROI_SNUG).sum())
        n_mild = int(points_in_roi(SMALL.x_nm, SMALL.y_nm, ROI_MILD).sum())
        assert 0.9 * SMALL.n_locs < n_snug < SMALL.n_locs, n_snug
        assert 0.95 * SMALL.n_locs < n_mild < SMALL.n_locs, n_mild
        touching = []
        for roi, axon in ((ROI_SNUG, EDGE_AXON), (ROI_MILD, MILD_AXON)):
            m = axon.ring == 1
            d = np.abs(distance_to_roi_boundary(axon.x_nm[m], axon.y_nm[m], roi))
            ids = axon.cluster[m]
            near = {int(c) for c in np.unique(ids) if np.any(d[ids == c] <= 25.0)}
            touching.append(len(near) / np.unique(ids).size)
        assert touching[0] > 0.5 and touching[1] < 0.5, touching
        return (f"snug ROI keeps {n_snug} of {SMALL.n_locs} localizations, {100 * touching[0]:.0f} % "
                f"of the middle ring's true clusters within 25 nm of it (> 50 %: guard fires); "
                f"mild ROI keeps {n_mild}, {100 * touching[1]:.0f} % touch (< 50 %: guard silent)")

    check("H2 API: bootstrap_centroid_sigma, tau_components, TauComponents, ColumnsParams, "
          "write/load_columns_params, rings_params_from; build_rings(n_frames=...)", api_present)
    check("data model: new Cluster/RingsParams/RingsResult fields with the spec's defaults; "
          "TauComponents fields", data_model)
    check("marks axon: burst implant in one twentieth (share 100 %); few-events implant of 12 "
          "consecutive frames straddling a window edge (share 50 %); DBSCAN(25, 10) keeps it", implants)
    check("SMALL: clusters failing Picasso's frame rule by chance (<= 8 of 72; none an implant)",
          chance_bursts)
    check("ROI truth: snug polygon touches > 50 % of the clusters, mild polygon < 50 %", roi_cuts)


# ============================================================ 1. bootstrap
def test_bootstrap() -> None:
    print("\n1. bootstrap_centroid_sigma: E = 30 events x k = 5 localizations, sigma_f = lp = 8 nm")
    st: Dict[str, Any] = {}
    sigma_f, lp, n_ev, k = 8.0, 8.0, 30, 5
    sigma_true = math.sqrt((sigma_f ** 2 + lp ** 2 / k) / n_ev)      # 1.60 nm
    n_real, n_draws = 300, 500

    def realisations():
        fn = require("bootstrap_centroid_sigma")
        rng = np.random.default_rng(0)
        seeds = np.random.SeedSequence(0).spawn(n_real)
        cents = np.empty((n_real, 2))
        boot_ev = np.empty(n_real)
        boot_loc = np.empty(n_real)
        for r in range(n_real):
            offsets = rng.normal(0.0, sigma_f, (n_ev, 1, 2))
            locs = (offsets + rng.normal(0.0, lp, (n_ev, k, 2))).reshape(-1, 2)
            x, y = locs[:, 0].copy(), locs[:, 1].copy()
            event_id = np.repeat(np.arange(n_ev, dtype=np.int64), k)
            cents[r] = x.mean(), y.mean()
            boot_ev[r] = fn(x, y, event_id, n_draws=n_draws,
                            rng=np.random.default_rng(seeds[r]))[0]
            boot_loc[r] = fn(x, y, np.arange(x.size, dtype=np.int64), n_draws=n_draws,
                             rng=np.random.default_rng(seeds[r]))[0]
        st["sd_emp"] = float(math.sqrt(cents.var(axis=0, ddof=1).mean()))
        st["boot_ev"] = float(boot_ev.mean())
        st["boot_loc"] = float(boot_loc.mean())
        return f"{n_real} realisations, {n_draws} draws each"

    def empirical_sd():
        # sd of a sd estimated from 300 x 2 axes = 600 values: 1/sqrt(2*599)
        # = 2.9 %; 10 % is 3.4 sd.
        sd = need(st, "sd_emp")
        assert abs(sd - sigma_true) / sigma_true < 0.10, sd
        return f"empirical sd {sd:.3f} nm vs sigma_true {sigma_true:.3f} nm"

    def event_bootstrap():
        # Bootstrap bias sqrt((E-1)/E) = 0.983 and the c4 of 500 draws are
        # within 2 %; the mean over 300 has sd ~ 0.5 %: 10 % is generous.
        sd, b = need(st, "sd_emp"), need(st, "boot_ev")
        assert abs(b - sd) / sd < 0.10, (b, sd)
        return f"mean bootstrap sigma over events {b:.3f} nm vs empirical {sd:.3f} nm ({b / sd:.3f})"

    def localization_bootstrap_understates():
        # Every localization its own event: sqrt((sigma_f^2 + lp^2) / (E k))
        # = 0.92 nm = 0.58 sigma_true; must be below 0.8 sigma_true.
        b = need(st, "boot_loc")
        assert b < 0.8 * sigma_true, b
        return (f"mean over localizations {b:.3f} nm = {b / sigma_true:.2f} sigma_true "
                f"(expected 0.58)")

    def small_e():
        fn = require("bootstrap_centroid_sigma")
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 8.0, 10)
        y = rng.normal(0.0, 8.0, 10)
        for ids, label in ((np.zeros(10, dtype=np.int64), "E = 1"),
                           (np.zeros(0, dtype=np.int64), "E = 0")):
            n = ids.size
            out = fn(x[:n], y[:n], ids, n_draws=100, rng=np.random.default_rng(0))
            assert len(out) == 3 and all(is_nan_float(float(v)) for v in out), (label, out)
        two = fn(x, y, np.repeat(np.arange(2, dtype=np.int64), 5), n_draws=100,
                 rng=np.random.default_rng(0))
        assert all(np.isfinite(two)) and two[0] > 0.0, two
        return f"E = 0, 1 -> (nan, nan, nan); E = 2 -> sigma {two[0]:.3f} nm (finite, > 0)"

    def minus_one_is_own_event():
        # 10 unlinked localizations of noise 8 nm: E = 10 singleton events.
        # Reference: the first-principles singleton bootstrap on the same
        # design (analytic ~ c4(10) sqrt(9/10) 8 / sqrt(10) = 2.33 nm); the
        # mean over 200 realisations has sd ~ 0.24 * 2.33 / sqrt(200) = 1.7 %
        # each side: 10 % is > 4 sd of the difference.
        fn = require("bootstrap_centroid_sigma")
        rng = np.random.default_rng(2)
        got = np.empty(200)
        ref = np.empty(200)
        for r in range(200):
            x = rng.normal(0.0, 8.0, 10)
            y = rng.normal(0.0, 8.0, 10)
            out = fn(x, y, np.full(10, -1, dtype=np.int64), n_draws=200,
                     rng=np.random.default_rng(1000 + r))
            assert all(np.isfinite(out)), out
            got[r] = out[0]
            ref[r] = bootstrap_sd_first_principles(np.column_stack([x, y]), 200,
                                                   np.random.default_rng(5000 + r))[0]
        assert abs(got.mean() - ref.mean()) / ref.mean() < 0.10, (got.mean(), ref.mean())
        return (f"ids all -1: finite; mean sigma {got.mean():.3f} nm vs singleton reference "
                f"{ref.mean():.3f} nm (analytic 2.33)")

    def mixed_ids():
        fn = require("bootstrap_centroid_sigma")
        rng = np.random.default_rng(3)
        x = rng.normal(0.0, 8.0, 10)
        y = rng.normal(0.0, 8.0, 10)
        ids = np.array([0, 0, 0, 0, 0, -1, -1, -1, -1, -1], dtype=np.int64)
        mixed = fn(x, y, ids, n_draws=200, rng=np.random.default_rng(0))
        assert all(np.isfinite(mixed)) and mixed[0] > 0.0, mixed
        one = fn(x, y, np.zeros(10, dtype=np.int64), n_draws=200, rng=np.random.default_rng(0))
        assert all(is_nan_float(float(v)) for v in one), one
        return f"5 linked + 5 unlinked (E = 6): sigma {mixed[0]:.3f} nm; all one event: NaN"

    def deterministic_and_consistent():
        fn = require("bootstrap_centroid_sigma")
        rng = np.random.default_rng(4)
        x = rng.normal(0.0, 8.0, 40)
        y = rng.normal(0.0, 8.0, 40)
        ids = np.repeat(np.arange(8, dtype=np.int64), 5)
        a = fn(x, y, ids, n_draws=300, rng=np.random.default_rng(11))
        b = fn(x, y, ids, n_draws=300, rng=np.random.default_rng(11))
        c = fn(x, y, ids, n_draws=300, rng=np.random.default_rng(12))
        assert tuple(map(float, a)) == tuple(map(float, b)), (a, b)
        assert tuple(map(float, a)) != tuple(map(float, c)), (a, c)
        for out in (a, c):
            s, sx, sy = map(float, out)
            assert abs(s - math.sqrt((sx * sx + sy * sy) / 2.0)) <= 1e-12 * max(s, 1e-300), out
        return (f"seed 11 twice: identical ({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}); seed 12 differs; "
                f"sigma == sqrt((sx^2 + sy^2) / 2) to 1e-12")

    def small_e_bias():
        # The plug-in bootstrap resamples the E observed event means, so
        # its sd of the mean is on average sqrt((E - 1) / E) of the truth,
        # and a little less (the mean of an sd is below the sd of the
        # mean: Jensen). The module documents the numbers instead of
        # correcting them (D-03 defines the estimator); this measures them
        # so they are not remembered. One localization per event, sd 10
        # nm per axis, truth 10 / sqrt(E); 2000 realisations x 200 draws.
        # Per realisation the bootstrap sigma scatters ~40-50 %, so the
        # mean over 2000 has sd ~1 %: each window below is >= 4 sd wide
        # on both sides of the measured 0.77 (E = 3), 0.83 (4), 0.98 (30).
        fn = require("bootstrap_centroid_sigma")
        ratios: Dict[int, float] = {}
        for n_e, lo, hi in ((3, 0.72, 0.82), (4, 0.78, 0.88), (30, 0.95, 1.01)):
            rng = np.random.default_rng(100 + n_e)
            seeds = np.random.SeedSequence(200 + n_e).spawn(2000)
            boots = np.empty(2000)
            for r in range(2000):
                x = rng.normal(0.0, 10.0, n_e)
                y = rng.normal(0.0, 10.0, n_e)
                boots[r] = fn(x, y, np.arange(n_e, dtype=np.int64), n_draws=200,
                              rng=np.random.default_rng(seeds[r]))[0]
            ratios[n_e] = float(boots.mean()) / (10.0 / math.sqrt(n_e))
            assert lo <= ratios[n_e] <= hi, (n_e, ratios[n_e], lo, hi)
        assert ratios[3] < ratios[4] < ratios[30]
        doc = require("bootstrap_centroid_sigma").__doc__ or ""
        assert "0.77" in doc and "0.83" in doc and "0.98" in doc, "bias numbers not in the docstring"
        return ("mean bootstrap sigma / truth: " + ", ".join(f"E = {k}: {v:.3f}" for k, v in ratios.items())
                + " (bias documented, not corrected)")

    check("300 realisations generated; bootstrap over events and over localizations computed",
          realisations)
    check("empirical sd of the 300 centroids within 10 % of sigma_true = 1.60 nm", empirical_sd)
    check("mean bootstrap sigma over EVENTS (500 draws) within 10 % of the empirical sd",
          event_bootstrap)
    check("bootstrap over LOCALIZATIONS understates: mean < 0.8 sigma_true", localization_bootstrap_understates)
    check("E = 0 or 1 -> NaN triple; E = 2 -> finite positive", small_e)
    check("event id -1 = its own event: 10 unlinked -> E = 10, finite; mean over 200 realisations "
          "within 10 % of the first-principles singleton bootstrap", minus_one_is_own_event)
    check("mixed ids: 5 linked + 5 unlinked = 6 events (finite); one event -> NaN", mixed_ids)
    check("determinism: same rng seed -> identical triple, other seed differs; "
          "sigma_nm == sqrt((sx^2 + sy^2) / 2) to 1e-12", deterministic_and_consistent)
    check("few events: the plug-in bootstrap understates the truth by the documented factor "
          "(E = 3: 0.72-0.82, E = 4: 0.78-0.88, E = 30: 0.95-1.01), monotone in E", small_e_bias)


# ============================================================ 2. SMALL baseline
def test_small_baseline() -> Dict[str, Any]:
    print("\n2. build_rings on SMALL with n_frames = 60000: sigma per cluster, hull path, H1 unchanged")
    st: Dict[str, Any] = {}
    axon = SMALL

    def plain_call_vs_fingerprint():
        # The H1 call of validate_rings_h1 (no n_frames): its H1 fields
        # must equal what the H1 code produced before H2 (fingerprint
        # file), floats to 1e-9 nm; warnings: the pre-H2 ones still present.
        fp = load_fingerprint()
        res = h1.run_build_rings(axon)
        st["plain"] = res
        now = h1_fingerprint(res)
        out: List[str] = []
        diff_json({k: v for k, v in fp.items() if k != "warnings"},
                  {k: v for k, v in now.items() if k != "warnings"}, "fp", out)
        assert not out, out
        missing = [w for w in fp["warnings"] if w not in now["warnings"]]
        assert not missing, missing
        n_cl = sum(len(r["clusters"]) for r in fp["rings"])
        return (f"{len(fp['rings'])} rings, {n_cl} clusters, frame, events identical to the "
                f"pre-H2 fingerprint; {len(fp['warnings'])} pre-H2 warnings all present "
                f"({len(now['warnings'])} now)")

    def runs():
        require_build_rings_kw("n_frames")
        res = h1.run_build_rings(axon, n_frames=N_FRAMES)
        st["res"] = res
        return (f"{len(res.rings)} rings, {sum(len(r.clusters) for r in res.rings)} clusters, "
                f"n_suspect {dict(res.n_suspect)}")

    def h1_fields_unchanged_with_n_frames():
        res, plain = need(st, "res"), need(st, "plain")
        out: List[str] = []
        a, b = h1_fingerprint(res), h1_fingerprint(plain)
        diff_json({k: v for k, v in a.items() if k != "warnings"},
                  {k: v for k, v in b.items() if k != "warnings"}, "fp", out)
        assert not out, out
        pairs = list(zip(all_clusters(res), all_clusters(plain)))
        for (_, ca), (_, cb) in pairs:
            for name in ("sigma_centroid_nm", "sigma_centroid_x_nm", "sigma_centroid_y_nm"):
                va, vb = float(getattr(ca, name)), float(getattr(cb, name))
                assert va == vb or (math.isnan(va) and math.isnan(vb)), (name, va, vb)
        return (f"H1 fields identical with and without n_frames; sigma identical over "
                f"{len(pairs)} clusters (determinism)")

    def sigma_per_cluster():
        res = need(st, "res")
        n = 0
        for _, cl in all_clusters(res):
            assert cl.n_events is not None and cl.n_events >= 2, cl.n_events
            s, sx, sy = (float(cl.sigma_centroid_nm), float(cl.sigma_centroid_x_nm),
                         float(cl.sigma_centroid_y_nm))
            assert np.isfinite([s, sx, sy]).all() and s > 0.0, (s, sx, sy)
            assert abs(s - math.sqrt((sx * sx + sy * sy) / 2.0)) <= 1e-12 * s, (s, sx, sy)
            assert cl.n_bootstrap == res.params.n_bootstrap == 200, cl.n_bootstrap
            n += 1
        return f"{n} clusters: sigma finite, == sqrt((sx^2 + sy^2) / 2), n_bootstrap 200"

    def sigma_magnitude():
        # A SMALL cluster is 4 events of 5 localizations, sigma_f = lp = 8:
        # true per-axis centroid sd sqrt((64 + 12.8) / 4) = 4.38 nm. The
        # bootstrap over E = 4 events is biased low (sqrt(3/4) and the c4 of
        # 4 event means, ~3.3-3.5 nm) and noisy per cluster (sd of an sd of
        # 4 values ~ 42 %), so the module's median over ~72 clusters (sd
        # 1.25 * 0.42 / sqrt(72) = 6 %) is compared with the first-
        # principles reference on the same design: 20 % is ~3 sd. The
        # localization-bootstrap reference (2.4 nm) lies outside that window.
        res = need(st, "res")
        vals = np.array([float(cl.sigma_centroid_nm) for _, cl in all_clusters(res)])
        med = float(np.median(vals))
        ref_ev = reference_median_sigma(0, n_clusters=2000, n_events=4, locs_per_event=5,
                                        sigma_f_nm=8.0, lp_nm=8.0,
                                        n_draws=int(res.params.n_bootstrap), singleton=False)
        ref_loc = reference_median_sigma(0, n_clusters=2000, n_events=4, locs_per_event=5,
                                         sigma_f_nm=8.0, lp_nm=8.0,
                                         n_draws=int(res.params.n_bootstrap), singleton=True)
        assert abs(med - ref_ev) / ref_ev < 0.20, (med, ref_ev)
        assert abs(ref_loc - ref_ev) / ref_ev > 0.20, "references not separated"
        return (f"median sigma {med:.2f} nm over {vals.size} clusters; event reference "
                f"{ref_ev:.2f} nm, localization reference {ref_loc:.2f} nm, truth 4.38 nm")

    def hull_path_marks():
        res = need(st, "res")
        chance = []
        for ring, cl in all_clusters(res):
            assert ring.analysis.edge_reference == "convex hull", ring.analysis.edge_reference
            assert ring.analysis.bad_report.edge_criterion_disabled is not None
            exp = expected_marks(axon, res, ring, cl, total_frames=N_FRAMES, roi=None,
                                 frames_available=True, min_events=res.params.min_events_per_cluster)
            assert "few_events" not in exp and "edge" not in exp
            assert marks_of(cl) == exp, (ring.index, cl.label, marks_of(cl), exp)
            if exp:
                chance.append((ring.index, cl.label, majority_true_cluster(axon, cl)))
        for m in MARKS:
            assert res.n_suspect.get(m, 0) == sum(m in marks_of(cl) for _, cl in all_clusters(res))
        return (f"guard fired on every ring (hull); no 'few_events', no 'edge'; 'burst' only on "
                f"the chance clusters (ring, label, true id) {chance}; n_suspect {dict(res.n_suspect)}")

    check("plain H1 call: H1 fields identical to the pre-H2 fingerprint (1e-9); pre-H2 warnings present",
          plain_call_vs_fingerprint)
    check("build_rings runs on SMALL with n_frames = 60000", runs)
    check("n_frames changes no H1 field; sigma identical between the two calls (determinism)",
          h1_fields_unchanged_with_n_frames)
    check("every cluster: sigma_centroid finite, sigma == sqrt((sx^2 + sy^2) / 2), n_bootstrap == 200",
          sigma_per_cluster)
    check("median sigma_centroid over SMALL clusters within 20 % of the first-principles event "
          "bootstrap (E = 4, k = 5); localization bootstrap excluded", sigma_magnitude)
    check("hull path: no 'edge' (nothing cut), no 'few_events'; 'burst' == frame rule (chance only); "
          "n_suspect matches", hull_path_marks)
    return st


# ============================================================ 3. marks
def test_marks() -> Dict[str, Any]:
    print("\n3. Suspect marks: implants on SMALL, absent frames, n_frames = None, a forwarded ROI")
    st: Dict[str, Any] = {}
    axon = MARKS_AXON

    def runs():
        require_build_rings_kw("n_frames")
        res = h1.run_build_rings(axon, n_frames=N_FRAMES)
        st["res"] = res
        assert len(res.rings) == 3, len(res.rings)
        rb, cb = find_cluster(axon, res, BURST_ID)
        rf, cf = find_cluster(axon, res, FEW_ID)
        st["burst"] = (rb, cb)
        st["few"] = (rf, cf)
        assert rb.index == rf.index == 1, (rb.index, rf.index)
        return (f"3 rings; burst implant kept as ring 1 label {cb.label} ({cb.n_locs} locs), "
                f"few-events implant as ring 1 label {cf.label} ({cf.n_locs} locs)")

    def burst_implant():
        _, cl = need(st, "burst")
        assert cl.n_events == 4, cl.n_events
        assert marks_of(cl) == {"burst"}, cl.suspect
        assert np.isfinite(float(cl.sigma_centroid_nm)), cl.sigma_centroid_nm
        return f"n_events 4, suspect {cl.suspect}, sigma {float(cl.sigma_centroid_nm):.2f} nm"

    def few_implant():
        _, cl = need(st, "few")
        assert cl.n_events is not None and cl.n_events <= 2, cl.n_events
        assert marks_of(cl) == {"few_events"}, cl.suspect
        s = float(cl.sigma_centroid_nm)
        if cl.n_events < 2:
            assert is_nan_float(s), s
        else:
            assert np.isfinite(s), s
        return f"n_events {cl.n_events}, suspect {cl.suspect}, sigma {s}"

    def others_as_expected():
        res = need(st, "res")
        _, cb = need(st, "burst")
        _, cf = need(st, "few")
        chance = []
        n_other = 0
        for ring, cl in all_clusters(res):
            exp = expected_marks(axon, res, ring, cl, total_frames=N_FRAMES, roi=None,
                                 frames_available=True, min_events=res.params.min_events_per_cluster)
            assert marks_of(cl) == exp, (ring.index, cl.label, marks_of(cl), exp)
            if cl is cb or cl is cf:
                continue
            n_other += 1
            assert "few_events" not in exp and "edge" not in exp, (ring.index, cl.label, exp)
            if exp:
                bursty, frac, share = frame_rule_bursty(
                    axon.frame[np.asarray(cl.loc_index, dtype=np.intp)], N_FRAMES)
                chance.append((ring.index, cl.label, round(frac, 3)))
        marked = sum(bool(marks_of(cl)) for _, cl in all_clusters(res))
        assert marked == 2 + len(chance), (marked, chance)
        return (f"{n_other} other clusters: marks == expectation; clean except {len(chance)} "
                f"chance bursts (ring, label, mean fraction) {chance}")

    def counts():
        res = need(st, "res")
        got = {m: sum(m in marks_of(cl) for _, cl in all_clusters(res)) for m in MARKS}
        for m in MARKS:
            assert res.n_suspect.get(m, 0) == got[m], (m, res.n_suspect, got)
        extra = {k: v for k, v in res.n_suspect.items() if k not in MARKS and v}
        assert not extra, extra
        assert got["burst"] >= 1 and got["few_events"] >= 1 and got["edge"] == 0, got
        return f"n_suspect {dict(res.n_suspect)} == counted {got}"

    def without_frames():
        require_build_rings_kw("n_frames")
        res = h1.run_build_rings(axon, frame=None, n_frames=N_FRAMES)
        n = 0
        for _, cl in all_clusters(res):
            assert marks_of(cl) == set(), cl.suspect
            assert is_nan_float(float(cl.sigma_centroid_nm)), cl.sigma_centroid_nm
            n += 1
        assert all(v == 0 for v in res.n_suspect.values()), res.n_suspect
        low = [w.lower() for w in res.warnings]
        sigma_w = [w for w in low if "frame" in w and any(k in w for k in ("sigma", "bootstrap", "precision"))]
        burst_w = [w for w in low if "frame" in w and any(k in w for k in ("burst", "mark", "suspect"))]
        assert sigma_w, res.warnings
        assert burst_w, res.warnings
        return (f"{n} clusters: no marks, sigma NaN; warnings: '{sigma_w[0][:60]}...' and "
                f"'{burst_w[0][:60]}...'")

    def n_frames_none():
        require_build_rings_kw("n_frames")
        res = h1.run_build_rings(axon, n_frames=None)
        st["res_none"] = res
        low = [w for w in res.warnings if "lower bound" in w.lower()]
        assert low, res.warnings
        _, cb = find_cluster(axon, res, BURST_ID)
        assert "burst" in marks_of(cb), cb.suspect
        total_all = int(axon.frame.max()) + 1
        variants = {"max over all frames + 1": lambda ring: total_all,
                    "max over the ring's frames + 1": lambda ring: int(
                        axon.frame[np.asarray(ring.loc_index, dtype=np.intp)].max()) + 1}
        matched = None
        for label, total_of in variants.items():
            ok = all(marks_of(cl) == expected_marks(
                axon, res, ring, cl, total_frames=total_of(ring), roi=None,
                frames_available=True, min_events=res.params.min_events_per_cluster)
                for ring, cl in all_clusters(res))
            if ok:
                matched = label
                break
        assert matched is not None, "marks match neither acquisition-length variant"
        _, cf = find_cluster(axon, res, FEW_ID)
        return (f"warning '{low[0][:70]}...'; burst implant marked; every mark == frame rule with "
                f"length = {matched} ({total_all}); few-events implant now {cf.suspect}")

    def edge_guard_fires():
        require_build_rings_kw("n_frames", "roi")
        eaxon = EDGE_AXON
        res = h1.run_build_rings(eaxon, params=h1.default_params(correct_tilt=False),
                                 n_frames=N_FRAMES, roi=ROI_SNUG)
        st["res_edge"] = res
        assert len(res.rings) == 3, len(res.rings)
        assert res.frame.is_identity
        per_ring = []
        for ring in res.rings:
            an = ring.analysis
            assert an.edge_reference == "roi", an.edge_reference
            assert an.bad_report.edge_criterion_disabled is not None, ring.index
            assert not an.bad_report.edge_touching
            n_edge = 0
            for cl in ring.clusters:
                exp = expected_marks(eaxon, res, ring, cl, total_frames=N_FRAMES, roi=ROI_SNUG,
                                     frames_available=True,
                                     min_events=res.params.min_events_per_cluster)
                assert marks_of(cl) == exp, (ring.index, cl.label, marks_of(cl), exp)
                n_edge += "edge" in exp
            assert 4 <= n_edge <= len(ring.clusters) - 4, (n_edge, len(ring.clusters))
            per_ring.append(f"{n_edge}/{len(ring.clusters)}")
        got = sum("edge" in marks_of(cl) for _, cl in all_clusters(res))
        assert res.n_suspect.get("edge", 0) == got, (res.n_suspect, got)
        return (f"guard fired on all 3 rings; 'edge' on the clusters within eps of the ROI "
                f"(per ring {per_ring}), none elsewhere; n_suspect['edge'] {got}")

    def edge_guard_silent():
        require_build_rings_kw("n_frames", "roi")
        maxon = MILD_AXON
        res = h1.run_build_rings(maxon, params=h1.default_params(correct_tilt=False),
                                 n_frames=N_FRAMES, roi=ROI_MILD)
        assert len(res.rings) == 3, len(res.rings)
        removed = []
        for ring in res.rings:
            an = ring.analysis
            assert an.edge_reference == "roi" and an.bad_report.edge_criterion_disabled is None
            assert an.bad_report.edge_touching, "no cluster touched the mild ROI"
            assert len(ring.clusters) == an.n_clusters_raw - len(an.bad_report.bad_labels)
            for cl in ring.clusters:
                assert cl.label not in an.bad_report.edge_touching
                assert "edge" not in marks_of(cl), (ring.index, cl.label, cl.suspect)
                li = np.asarray(cl.loc_index, dtype=np.intp)
                d = distance_to_roi_boundary(res.x_p[li], res.y_p[li], ROI_MILD)
                assert not np.any(np.abs(d) <= an.eps_nm), (ring.index, cl.label)
            removed.append(len(an.bad_report.edge_touching))
        assert res.n_suspect.get("edge", 0) == 0, res.n_suspect
        return (f"guard silent; touching clusters removed by H1's criterion (per ring {removed}); "
                f"no kept cluster within eps of the ROI, no 'edge' mark")

    check("build_rings on the marks axon (n_frames = 60000): both implants kept in ring 1", runs)
    check("(a) burst implant: 4 events, suspect == ('burst',) exactly, sigma finite", burst_implant)
    check("(b) few-events implant: n_events <= 2, suspect == ('few_events',) exactly, sigma NaN with 1 event",
          few_implant)
    check("(c) every other cluster: marks == first-principles expectation (clean except the chance bursts)",
          others_as_expected)
    check("(d) n_suspect counts per mark equal the clusters' marks; no other key", counts)
    check("(e) without frame: no marks, sigma NaN, warnings mention the bootstrap and the burst mark",
          without_frames)
    check("(f) n_frames = None: 'lower bound' warning, burst implant still marked, marks == rule with max(frame) + 1",
          n_frames_none)
    check("edge: ROI forwarded (snug on 2/3 of the ring, correct_tilt False): guard fires, kept clusters "
          "within eps carry 'edge', the others do not", edge_guard_fires)
    check("edge control: ROI snug on 1/4: guard silent, touching clusters removed (H1), no 'edge' mark",
          edge_guard_silent)
    return st


# ============================================================ 4. tau
def usable_cluster(cl: Any, exclude_suspect: bool) -> bool:
    return (np.isfinite(float(cl.rho_nm)) and np.isfinite(float(cl.sigma_centroid_nm))
            and (not exclude_suspect or not marks_of(cl)))


def recompute_tau(results: Sequence[Any], *, sigma_d_nm: float,
                  exclude_suspect: bool) -> Dict[str, Any]:
    """tau_0 and its components from the definition in the H2 spec: for
    every pair of consecutive rings (by ``Ring.index``) of every axon and
    every cluster i of ring k and j of ring k + 1 (full cross product),
    t_ij = rho_i + rho_j + 2 sqrt(sigma_m_i^2 + sigma_m_j^2 + shift^2 +
    sigma_d^2) with shift = period * tan(delta_beta) of that axon (0
    when delta_beta is NaN) and period the median over the listed rings
    of centre spacing / index step (a missing segment is a step of 2);
    tau_0 is the median of every t_ij."""
    t_all: List[float] = []
    rp_all: List[float] = []
    sp_all: List[float] = []
    rho_used: List[float] = []
    sm_used: List[float] = []
    n_used = n_exc = 0
    shifts: List[float] = []
    periods: List[float] = []
    for res in results:
        rings = list(res.rings)
        centres = np.array([r.centre_z_nm for r in rings], dtype=float)
        steps = np.diff(np.array([int(r.index) for r in rings], dtype=float))
        if centres.size >= 2 and not np.any(steps == 0.0):
            period = float(np.median(np.diff(centres) / steps))
        else:
            period = float("nan")
        db = float(res.frame.delta_beta_deg)
        shift = period * math.tan(math.radians(db)) if (np.isfinite(db) and np.isfinite(period)) else 0.0
        periods.append(period)
        shifts.append(shift)
        for ring in rings:
            for cl in ring.clusters:
                if usable_cluster(cl, exclude_suspect):
                    n_used += 1
                    rho_used.append(float(cl.rho_nm))
                    sm_used.append(float(cl.sigma_centroid_nm))
                else:
                    n_exc += 1
        for k in range(len(rings) - 1):
            if int(rings[k + 1].index) != int(rings[k].index) + 1:
                continue
            a = [cl for cl in rings[k].clusters if usable_cluster(cl, exclude_suspect)]
            b = [cl for cl in rings[k + 1].clusters if usable_cluster(cl, exclude_suspect)]
            for ci in a:
                for cj in b:
                    sp = math.sqrt(float(ci.sigma_centroid_nm) ** 2 + float(cj.sigma_centroid_nm) ** 2
                                   + shift * shift + sigma_d_nm * sigma_d_nm)
                    rp = float(ci.rho_nm) + float(cj.rho_nm)
                    t_all.append(rp + 2.0 * sp)
                    rp_all.append(rp)
                    sp_all.append(sp)
    t = np.asarray(t_all)
    q = [10, 25, 50, 75, 90]
    return {
        "t": t, "rp": np.asarray(rp_all), "sp": np.asarray(sp_all),
        "tau0": float(np.median(t)) if t.size else float("nan"),
        "rp_med": float(np.median(rp_all)) if rp_all else float("nan"),
        "sp_med": float(np.median(sp_all)) if sp_all else float("nan"),
        "sm_med": float(np.median(sm_used)) if sm_used else float("nan"),
        "n_pairs": int(t.size), "n_used": n_used, "n_exc": n_exc,
        "rho_q": np.percentile(rho_used, q) if rho_used else np.full(5, np.nan),
        "sm_q": np.percentile(sm_used, q) if sm_used else np.full(5, np.nan),
        "period": periods[0] if len(periods) == 1 else float("nan"),
        "shift": shifts[0] if len(shifts) == 1 else float("nan"),
    }


def test_tau(small_st: Dict[str, Any], marks_st: Dict[str, Any]) -> None:
    print("\n4. tau_components on the SMALL result (and the marks result, pooled, truncated)")
    st: Dict[str, Any] = {}

    def small_result() -> Any:
        if "res" not in small_st:
            raise RuntimeError("the SMALL baseline result is unavailable (section 2 failed)")
        return small_st["res"]

    def marks_result() -> Any:
        if "res" not in marks_st:
            raise RuntimeError("the marks result is unavailable (section 3 failed)")
        return marks_st["res"]

    def medians_match():
        fn = require("tau_components")
        res = small_result()
        tc = fn([res])
        st["tc"] = tc
        ref = recompute_tau([res], sigma_d_nm=0.0, exclude_suspect=True)
        st["ref"] = ref
        assert ref["n_pairs"] > 0
        assert abs(float(tc.tau0_nm) - ref["tau0"]) <= 1e-9, (tc.tau0_nm, ref["tau0"])
        assert abs(float(tc.rho_pair_median_nm) - ref["rp_med"]) <= 1e-9, (tc.rho_pair_median_nm, ref["rp_med"])
        assert abs(float(tc.sigma_pair_median_nm) - ref["sp_med"]) <= 1e-9, (tc.sigma_pair_median_nm, ref["sp_med"])
        assert abs(float(tc.sigma_m_median_nm) - ref["sm_med"]) <= 1e-9, (tc.sigma_m_median_nm, ref["sm_med"])
        assert int(tc.n_pairs) == ref["n_pairs"], (tc.n_pairs, ref["n_pairs"])
        assert int(tc.n_axons) == 1
        assert float(tc.sigma_d_nm) == 0.0
        return (f"tau0 {tc.tau0_nm:.3f} nm over {tc.n_pairs} pairs; rho_pair median "
                f"{tc.rho_pair_median_nm:.3f}, sigma_pair median {tc.sigma_pair_median_nm:.3f}, "
                f"sigma_m median {tc.sigma_m_median_nm:.3f} nm")

    def counts_and_quantiles():
        tc, ref = need(st, "tc"), need(st, "ref")
        assert int(tc.n_clusters_used) == ref["n_used"], (tc.n_clusters_used, ref["n_used"])
        assert int(tc.n_clusters_excluded) == ref["n_exc"], (tc.n_clusters_excluded, ref["n_exc"])
        rq = np.asarray(tc.rho_quantiles_nm, dtype=float)
        sq = np.asarray(tc.sigma_m_quantiles_nm, dtype=float)
        assert rq.shape == (5,) and np.all(np.abs(rq - ref["rho_q"]) <= 1e-9), (rq, ref["rho_q"])
        assert sq.shape == (5,) and np.all(np.abs(sq - ref["sm_q"]) <= 1e-9), (sq, ref["sm_q"])
        return (f"used {tc.n_clusters_used}, excluded {tc.n_clusters_excluded}; rho quantiles "
                f"{np.round(rq, 2).tolist()}, sigma_m quantiles {np.round(sq, 2).tolist()} nm")

    def shift_field():
        tc, ref = need(st, "tc"), need(st, "ref")
        res = small_result()
        period, db = float(tc.period_nm), float(tc.delta_beta_deg)
        assert abs(period - ref["period"]) <= 1e-9, (period, ref["period"])
        assert db == float(res.frame.delta_beta_deg) or (math.isnan(db) and math.isnan(res.frame.delta_beta_deg))
        expected = period * math.tan(math.radians(db))
        shift = float(tc.lateral_shift_nm)
        assert abs(shift - expected) <= 1e-9 or (math.isnan(shift) and math.isnan(expected)), (shift, expected)
        return f"period {period:.3f} nm, delta_beta {db:.4f} deg, lateral shift {shift:.4f} nm"

    def sigma_d_raises():
        # 2 (sqrt(s^2 + 400) - s) is in (0, 40] for every pair, so the
        # medians of the shifted arrays move by an amount in [0, 40].
        fn = require("tau_components")
        res = small_result()
        tc0 = need(st, "tc")
        tc20 = fn([res], sigma_d_nm=20.0)
        ref0 = need(st, "ref")
        ref20 = recompute_tau([res], sigma_d_nm=20.0, exclude_suspect=True)
        d_pairs = ref20["t"] - ref0["t"]
        assert np.all(d_pairs > 0.0) and np.all(d_pairs <= 40.0 + 1e-9), (d_pairs.min(), d_pairs.max())
        d_tau = float(tc20.tau0_nm) - float(tc0.tau0_nm)
        assert 0.0 <= d_tau <= 40.0, d_tau
        assert abs(float(tc20.tau0_nm) - ref20["tau0"]) <= 1e-9
        assert float(tc20.sigma_d_nm) == 20.0
        d_sp = float(tc20.sigma_pair_median_nm) - float(tc0.sigma_pair_median_nm)
        assert 0.0 <= d_sp <= 20.0, d_sp
        return (f"tau0 {tc0.tau0_nm:.3f} -> {tc20.tau0_nm:.3f} nm (+{d_tau:.3f}); pair tolerances "
                f"+{d_pairs.min():.3f}..{d_pairs.max():.3f} nm; sigma_pair median +{d_sp:.3f}")

    def include_suspects():
        fn = require("tau_components")
        res = marks_result()
        with_ex = fn([res], exclude_suspect=True)
        without = fn([res], exclude_suspect=False)
        ref = recompute_tau([res], sigma_d_nm=0.0, exclude_suspect=False)
        assert int(without.n_clusters_used) > int(with_ex.n_clusters_used), (without.n_clusters_used, with_ex.n_clusters_used)
        assert int(without.n_clusters_excluded) < int(with_ex.n_clusters_excluded)
        assert int(without.n_pairs) > int(with_ex.n_pairs)
        assert abs(float(without.tau0_nm) - ref["tau0"]) <= 1e-9, (without.tau0_nm, ref["tau0"])
        assert int(without.n_clusters_used) == ref["n_used"] and int(without.n_pairs) == ref["n_pairs"]
        return (f"exclude_suspect True: {with_ex.n_clusters_used} clusters, {with_ex.n_pairs} pairs, "
                f"tau0 {with_ex.tau0_nm:.3f}; False: {without.n_clusters_used} clusters, "
                f"{without.n_pairs} pairs, tau0 {without.tau0_nm:.3f} nm (== recomputation)")

    def pooled():
        fn = require("tau_components")
        a, b = small_result(), marks_result()
        tc = fn([a, b])
        ref = recompute_tau([a, b], sigma_d_nm=0.0, exclude_suspect=True)
        ra = recompute_tau([a], sigma_d_nm=0.0, exclude_suspect=True)
        rb = recompute_tau([b], sigma_d_nm=0.0, exclude_suspect=True)
        assert int(tc.n_axons) == 2
        assert int(tc.n_pairs) == ra["n_pairs"] + rb["n_pairs"] == ref["n_pairs"]
        assert abs(float(tc.tau0_nm) - ref["tau0"]) <= 1e-9, (tc.tau0_nm, ref["tau0"])
        assert abs(float(tc.tau0_nm) - float(np.median(np.concatenate([ra["t"], rb["t"]])))) <= 1e-9
        assert int(tc.n_clusters_used) == ra["n_used"] + rb["n_used"]
        return (f"2 axons, {tc.n_pairs} pairs ({ra['n_pairs']} + {rb['n_pairs']}), tau0 "
                f"{tc.tau0_nm:.3f} nm == median over the union")

    def one_ring():
        fn = require("tau_components")
        res = small_result()
        # Rings AND segments truncated, whichever the period is read from.
        ms_one = dataclasses.replace(res.ms, segments=list(res.ms.segments[:1]),
                                     analyses=list(res.ms.analyses[:1]), pairs=[])
        one = dataclasses.replace(res, rings=list(res.rings[:1]), ms=ms_one)
        tc = fn([one])
        for name in ("tau0_nm", "period_nm", "rho_pair_median_nm", "sigma_pair_median_nm"):
            assert is_nan_float(float(getattr(tc, name))), (name, getattr(tc, name))
        assert int(tc.n_pairs) == 0
        assert tc.warnings, "no warning for a single ring"
        return f"1 ring: tau0/period/pair medians NaN, n_pairs 0; warning '{tc.warnings[0][:70]}'"

    def deterministic():
        fn = require("tau_components")
        res = small_result()
        a, b = fn([res], sigma_d_nm=5.0), fn([res], sigma_d_nm=5.0)
        for f in dataclasses.fields(a):
            va, vb = getattr(a, f.name), getattr(b, f.name)
            if isinstance(va, np.ndarray):
                assert np.array_equal(va, vb, equal_nan=True), f.name
            elif isinstance(va, float):
                assert va == vb or (math.isnan(va) and math.isnan(vb)), (f.name, va, vb)
            else:
                assert va == vb, (f.name, va, vb)
        return f"two calls identical on every field ({len(dataclasses.fields(a))} fields)"

    def period_per_index_step():
        # A missing segment leaves a gap in Ring.index. The period must be
        # the spacing per index step: (a) rings 0 and 2 listed -> one step
        # of 2, period = (c2 - c0) / 2, which equals the 3-ring median of
        # (d1, d2) = their mean; no pairs, with the warning. (b) rings 0,
        # 1, 3 with ring 3 = ring 2 moved up one period p: the steps are
        # d1 and (c2 + p - c1) / 2, both within a few nm of p, so the
        # period stays between them (< 1.05 p) where the median over the
        # LISTED rings would be median(d1, c2 + p - c1) ~ 1.5 p; the 552
        # pairs of rings 0-1 and tau0 equal the recomputation. (c) a
        # repeated index -> NaN period, warning, no pairs.
        fn = require("tau_components")
        res = small_result()
        rings = list(res.rings)
        assert [int(r.index) for r in rings] == [0, 1, 2], [r.index for r in rings]
        c = [float(r.centre_z_nm) for r in rings]
        full = fn([res])
        p = float(full.period_nm)
        gap = dataclasses.replace(res, rings=[rings[0], rings[2]])
        tg = fn([gap])
        assert abs(float(tg.period_nm) - (c[2] - c[0]) / 2.0) <= 1e-9, (tg.period_nm, (c[2] - c[0]) / 2.0)
        assert abs(float(tg.period_nm) - p) <= 1e-9, (tg.period_nm, p)
        assert int(tg.n_pairs) == 0 and is_nan_float(float(tg.tau0_nm))
        assert any("not consecutive" in w for w in tg.warnings), tg.warnings
        r3 = dataclasses.replace(rings[2], index=3, centre_z_nm=c[2] + p)
        four = dataclasses.replace(res, rings=[rings[0], rings[1], r3])
        t4 = fn([four])
        ref4 = recompute_tau([four], sigma_d_nm=0.0, exclude_suspect=True)
        steps = [c[1] - c[0], (c[2] + p - c[1]) / 2.0]
        assert abs(float(t4.period_nm) - float(np.median(steps))) <= 1e-9, (t4.period_nm, steps)
        assert abs(float(t4.period_nm) - ref4["period"]) <= 1e-9
        assert min(steps) - 1e-9 <= float(t4.period_nm) <= max(steps) + 1e-9
        assert float(t4.period_nm) < 1.05 * p, (t4.period_nm, p)
        listed_only = float(np.median([c[1] - c[0], c[2] + p - c[1]]))
        assert listed_only > 1.4 * p, (listed_only, p)
        two = fn([dataclasses.replace(res, rings=rings[:2])])
        assert int(t4.n_pairs) == int(two.n_pairs) == ref4["n_pairs"] > 0, (t4.n_pairs, two.n_pairs)
        assert abs(float(t4.tau0_nm) - ref4["tau0"]) <= 1e-9, (t4.tau0_nm, ref4["tau0"])
        expected_shift = float(t4.period_nm) * math.tan(math.radians(float(t4.delta_beta_deg)))
        assert abs(float(t4.lateral_shift_nm) - expected_shift) <= 1e-9
        assert any("not consecutive" in w for w in t4.warnings), t4.warnings
        dup = dataclasses.replace(res, rings=[rings[0], dataclasses.replace(rings[1], index=0), rings[2]])
        td = fn([dup])
        assert is_nan_float(float(td.period_nm)) and int(td.n_pairs) == 0
        assert any("repeated index" in w for w in td.warnings), td.warnings
        return (f"rings [0, 2]: period {tg.period_nm:.3f} == 3-ring {p:.3f} nm, 0 pairs; rings [0, 1, 3]: "
                f"period {t4.period_nm:.3f} nm (listed-only median would be {listed_only:.1f}), "
                f"{t4.n_pairs} pairs, tau0 {t4.tau0_nm:.4f} == recomputation; repeated index: NaN")

    check("tau0 == median of the recomputed t_ij (cross product of consecutive rings, suspects out) "
          "to 1e-9; rho_pair/sigma_pair/sigma_m medians likewise; n_pairs == count", medians_match)
    check("n_clusters_used/excluded == usable / not; rho and sigma_m quantiles == np.percentile 10/25/50/75/90",
          counts_and_quantiles)
    check("period == median ring spacing; delta_beta from the frame; lateral_shift == period * tan(delta_beta)",
          shift_field)
    check("sigma_d_nm = 20: tau0 and every pair tolerance rise by >= 0 and <= 2 * 20 nm", sigma_d_raises)
    check("exclude_suspect=False on the marks axon uses strictly more clusters/pairs and equals the recomputation",
          include_suspects)
    check("two results pooled: n_pairs adds up, tau0 == median over the union of both axons' t_ij", pooled)
    check("one ring only: tau0, period and pair medians NaN, n_pairs 0, with a warning", one_ring)
    check("determinism: two tau_components calls identical on every field", deterministic)
    check("period per Ring.index step: rings [0, 2] give the 3-ring period and no pairs; rings [0, 1, 3] "
          "give a period between the per-step spacings (not the listed-only 1.5x) and tau0 == recomputation; "
          "a repeated index gives NaN with a warning", period_per_index_step)


# ============================================================ 5. ColumnsParams
def test_columns_params() -> None:
    print("\n5. ColumnsParams: YAML round-trip, strictness, readability, rings_params_from")
    st: Dict[str, Any] = {}
    tmp_dir = tempfile.mkdtemp(prefix="mps_h2_params_")

    def make_params() -> Any:
        cp = require("ColumnsParams")
        return cp(
            tau0_nm=63.2,
            tau_grid_nm=[float(v) for v in range(20, 201, 10)],
            tau_sensitivity_nm=[30.0, 63.2, 100.0],
            n_null=999, random_seed=3, min_shift_fraction=0.4, guard_nm=10.0,
            posterior_min=0.85, segment_mode="paper", eps_nm=30.0, min_samples=12,
            link_radius_factor=4.0, link_max_dark_time=2, n_bootstrap=50,
            min_events_per_cluster=4, sigma_d_nm=0.0,
            provenance={"dataset_role": "exploratory",
                        "source_files": ["exploratory/axon.hdf5"],
                        "date": "2026-09-23", "code_commit": "8f39e0a",
                        "tau_components": {"tau0_nm": 63.2, "n_pairs": 1234, "n_axons": 1}},
        )

    def round_trip():
        write, load = require("write_columns_params", "load_columns_params")
        params = make_params()
        path = os.path.join(tmp_dir, "columns_params.yaml")
        write(path, params)
        back = load(path)
        st["path"] = path
        st["params"] = params
        assert dataclasses.asdict(back) == dataclasses.asdict(params), (
            dataclasses.asdict(back), dataclasses.asdict(params))
        assert back == params
        return f"{len(dataclasses.fields(params))} fields equal after write -> load ({path})"

    def types_preserved():
        load = require("load_columns_params")
        back = load(need(st, "path"))
        for name in ("tau0_nm", "min_shift_fraction", "guard_nm", "eps_nm", "link_radius_factor",
                     "sigma_d_nm", "posterior_min"):
            assert type(getattr(back, name)) is float, (name, type(getattr(back, name)))
        for name in ("n_null", "random_seed", "min_samples", "link_max_dark_time", "n_bootstrap",
                     "min_events_per_cluster"):
            assert type(getattr(back, name)) is int, (name, type(getattr(back, name)))
        assert all(type(v) is float for v in back.tau_grid_nm), back.tau_grid_nm
        assert all(type(v) is float for v in back.tau_sensitivity_nm)
        assert isinstance(back.provenance, dict) and back.provenance["tau_components"]["n_pairs"] == 1234
        # None must survive as None (posterior_min is the primary path's default).
        write = require("write_columns_params")
        p2 = dataclasses.replace(need(st, "params"), posterior_min=None, guard_nm=0.0)
        path2 = os.path.join(tmp_dir, "columns_params_none.yaml")
        write(path2, p2)
        b2 = load(path2)
        assert b2.posterior_min is None and type(b2.guard_nm) is float and b2.guard_nm == 0.0
        assert b2 == p2
        return "floats float (0.0 stays float), ints int, lists of float, None round-trips, provenance dict"

    def human_readable():
        # PyYAML 6.0.3 is installed; its stubs (types-PyYAML) are not.
        import yaml  # type: ignore[import-untyped]
        cp = require("ColumnsParams")
        with open(need(st, "path"), "r", encoding="utf-8") as fh:
            text = fh.read()
        lines = text.splitlines()
        assert lines and lines[0].startswith("#"), lines[:2]
        n_header = 0
        for line in lines:
            if line.startswith("#"):
                n_header += 1
            else:
                break
        assert n_header >= 1
        assert "!!python" not in text and "!!" not in text, "python-specific YAML tags"
        keys = [m.group(1) for m in (re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", ln) for ln in lines) if m]
        expected = [f.name for f in dataclasses.fields(cp)]
        assert keys == expected, (keys, expected)
        raw = yaml.safe_load(text)
        assert isinstance(raw, dict) and set(raw) == set(expected)
        assert re.search(r"^tau0_nm:\s*63\.2\s*$", text, re.M), "tau0_nm line not plain"
        return f"{n_header} header comment lines, {len(keys)} top-level keys in dataclass order, safe_load ok"

    def strictness():
        load = require("load_columns_params")
        base = ("# pre-specified parameters (test)\n"
                "tau0_nm: 63.2\n"
                "tau_grid_nm: [20.0, 30.0]\n"
                "tau_sensitivity_nm: [30.0, 63.2, 100.0]\n")
        bad = os.path.join(tmp_dir, "unknown_key.yaml")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write(base + "tau_grd_nm: [1.0]\n")
        try:
            load(bad)
        except ValueError as exc:
            assert "tau_grd_nm" in str(exc), exc
        else:
            raise AssertionError("unknown key accepted")
        missing = os.path.join(tmp_dir, "missing_tau0.yaml")
        with open(missing, "w", encoding="utf-8") as fh:
            fh.write("tau_grid_nm: [20.0, 30.0]\ntau_sensitivity_nm: [30.0, 63.2, 100.0]\n")
        try:
            load(missing)
        except ValueError as exc:
            assert "tau0" in str(exc), exc
        else:
            raise AssertionError("missing tau0_nm accepted")
        minimal = os.path.join(tmp_dir, "minimal.yaml")
        with open(minimal, "w", encoding="utf-8") as fh:
            fh.write(base)
        cp = require("ColumnsParams")
        got = load(minimal)
        defaults = {f.name: f.default for f in dataclasses.fields(cp)
                    if f.default is not dataclasses.MISSING}
        for name, value in defaults.items():
            assert getattr(got, name) == value, (name, getattr(got, name), value)
        assert got.provenance == {}
        return ("unknown key 'tau_grd_nm' -> ValueError naming it; missing tau0_nm -> ValueError; "
                f"minimal file loads with {len(defaults)} defaults")

    def rings_params():
        fn, cp = require("rings_params_from", "ColumnsParams")
        params = make_params()
        rp = fn(params)
        assert isinstance(rp, RingsParams), type(rp)
        shared = sorted({f.name for f in dataclasses.fields(cp)}
                        & {f.name for f in dataclasses.fields(RingsParams)})
        # The fields the spec lists in both classes; make_params gives each
        # a value that differs from the RingsParams default, so "copied"
        # is distinguishable from "left at the default".
        spec_shared = {"segment_mode", "guard_nm", "eps_nm", "min_samples", "link_radius_factor",
                       "link_max_dark_time", "posterior_min", "random_seed", "n_bootstrap",
                       "min_events_per_cluster"}
        assert spec_shared <= set(shared), sorted(spec_shared - set(shared))
        default = RingsParams()
        for name in shared:
            assert getattr(rp, name) == getattr(params, name), (name, getattr(rp, name), getattr(params, name))
            if name in spec_shared:
                assert getattr(params, name) != getattr(default, name), f"{name}: test value equals the default"
        for f in dataclasses.fields(RingsParams):
            if f.name not in shared:
                assert getattr(rp, f.name) == getattr(default, f.name), f.name
        return f"shared fields copied exactly: {shared}; the others at RingsParams defaults"

    BASE = ("# pre-specified parameters (test)\n"
            "tau0_nm: 63.2\n"
            "tau_grid_nm: [20.0, 30.0]\n"
            "tau_sensitivity_nm: [30.0, 63.2, 100.0]\n")

    def expect_value_error(load: Any, text: str, key: str, label: str) -> str:
        path = os.path.join(tmp_dir, "bad.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        try:
            load(path)
        except ValueError as exc:
            assert key in str(exc), (label, str(exc))
            return str(exc)
        raise AssertionError(f"{label}: accepted")

    def duplicate_keys():
        # PyYAML keeps the last value of a repeated key silently; the
        # loader must name the key, at the top level and inside the
        # provenance, in block and in flow style. A repeated VALUE is
        # not a repeated key.
        load = require("load_columns_params")
        cases = (
            ("n_null twice", BASE + "n_null: 5\nn_null: 6\n", "n_null"),
            ("tau0_nm twice", "tau0_nm: 40.0\n" + BASE + "tau0_nm: 99.0\n", "tau0_nm"),
            ("provenance date twice (block)",
             BASE + "provenance:\n  dataset_role: exploratory\n  date: a\n  date: b\n", "date"),
            ("provenance date twice (flow)",
             BASE + "provenance: {date: a, code_commit: x, date: b}\n", "date"),
        )
        for label, text, key in cases:
            msg = expect_value_error(load, text, key, label)
            assert "twice" in msg, msg
        path = os.path.join(tmp_dir, "same_value.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(BASE + "provenance: {dataset_role: exploratory, date: '8f39e0a', code_commit: '8f39e0a'}\n")
        got = load(path)
        assert got.provenance["date"] == got.provenance["code_commit"] == "8f39e0a"
        return f"{len(cases)} repeated keys -> ValueError naming the key; a repeated value loads"

    def ranges():
        # Values of the right type that cannot mean anything must fail on
        # load, on write and in columns_params_from_rings, naming the
        # field; the ends of each range are accepted.
        load, write = require("load_columns_params", "write_columns_params")
        from_rings = require("columns_params_from_rings")
        rejected = (
            ("n_null: 0\n", "n_null"), ("n_null: -5\n", "n_null"), ("min_samples: 0\n", "min_samples"),
            ("eps_nm: -25.0\n", "eps_nm"), ("eps_nm: 0.0\n", "eps_nm"),
            ("posterior_min: 1.5\n", "posterior_min"), ("posterior_min: -0.1\n", "posterior_min"),
            ("segment_mode: valey\n", "segment_mode"), ("n_bootstrap: -1\n", "n_bootstrap"),
            ("min_events_per_cluster: 0\n", "min_events_per_cluster"),
            ("sigma_d_nm: -1.0\n", "sigma_d_nm"), ("guard_nm: -1.0\n", "guard_nm"),
            ("min_shift_fraction: -0.5\n", "min_shift_fraction"),
            ("link_radius_factor: 0.0\n", "link_radius_factor"), ("link_max_dark_time: -1\n", "link_max_dark_time"),
        )
        for extra, key in rejected:
            expect_value_error(load, BASE + extra, key, extra.strip())
        lists = "tau_grid_nm: [20.0, 30.0]\ntau_sensitivity_nm: [30.0, 63.2, 100.0]\n"
        for text, key in (("tau0_nm: -1.0\n" + lists, "tau0_nm"), ("tau0_nm: 0.0\n" + lists, "tau0_nm"),
                          ("tau0_nm: 63.2\ntau_grid_nm: []\ntau_sensitivity_nm: [30.0]\n", "tau_grid_nm"),
                          ("tau0_nm: 63.2\ntau_grid_nm: [20.0, -30.0]\ntau_sensitivity_nm: [30.0]\n", "tau_grid_nm"),
                          ("tau0_nm: 63.2\ntau_grid_nm: [20.0]\ntau_sensitivity_nm: [0.0]\n", "tau_sensitivity_nm")):
            expect_value_error(load, text, key, key)
        accepted = ("posterior_min: 0.0\n", "posterior_min: 1.0\n", "n_bootstrap: 0\n", "guard_nm: 0.0\n",
                    "sigma_d_nm: 0.0\n", "min_shift_fraction: 0.0\n", "segment_mode: paper\n",
                    "segment_mode: partition\n", "link_max_dark_time: 0\n", "n_null: 1\n", "min_samples: 1\n")
        ok = os.path.join(tmp_dir, "edge_ok.yaml")
        for extra in accepted:
            with open(ok, "w", encoding="utf-8") as fh:
                fh.write(BASE + extra)
            load(ok)
        params = make_params()
        refused = os.path.join(tmp_dir, "refused.yaml")
        for field, value in (("eps_nm", -1.0), ("n_null", 0), ("segment_mode", "valey"), ("posterior_min", 2.0)):
            try:
                write(refused, dataclasses.replace(params, **{field: value}))
            except ValueError as exc:
                assert field in str(exc), (field, str(exc))
            else:
                raise AssertionError(f"write accepted {field} = {value!r}")
        assert not os.path.exists(refused), "a refused write left a file behind"
        for bad_tau in (-1.0, 0.0):
            try:
                from_rings(bad_tau, provenance=params.provenance)
            except ValueError as exc:
                assert "tau0_nm" in str(exc), str(exc)
            else:
                raise AssertionError(f"columns_params_from_rings accepted tau0 {bad_tau}")
        return (f"{len(rejected) + 5} out-of-range files -> ValueError naming the field; "
                f"{len(accepted)} boundary values load; write and columns_params_from_rings refuse the same")

    def provenance_required():
        # 03_plan S3.1: dataset_role is one of three roles; a frozen file
        # without role, date or commit is not a pre-specification. The
        # writer refuses it (and leaves no file); the loader does not
        # require it (the minimal file of `strictness` loads with {}).
        write, load = require("write_columns_params", "load_columns_params")
        roles, keys = require("DATASET_ROLES", "PROVENANCE_REQUIRED_KEYS")
        assert tuple(roles) == ("exploratory", "check", "confirmatory"), roles
        assert set(keys) == {"dataset_role", "date", "code_commit"}, keys
        params = make_params()
        full = dict(params.provenance)
        path = os.path.join(tmp_dir, "prov.yaml")
        bad = (
            ({}, "dataset_role"),
            ({k: v for k, v in full.items() if k != "date"}, "date"),
            ({k: v for k, v in full.items() if k != "code_commit"}, "code_commit"),
            ({**full, "dataset_role": "explorat"}, "dataset_role"),
            ({**full, "date": ""}, "date"),
            ({**full, "code_commit": None}, "code_commit"),
        )
        for prov, key in bad:
            try:
                write(path, dataclasses.replace(params, provenance=prov))
            except ValueError as exc:
                assert key in str(exc), (prov, str(exc))
            else:
                raise AssertionError(f"write accepted provenance {prov!r}")
            assert not os.path.exists(path), "a refused write left a file behind"
        for role in roles:
            prov = {"dataset_role": role, "date": "2026-09-23", "code_commit": "8f39e0a"}
            write(path, dataclasses.replace(params, provenance=prov))
            assert load(path).provenance == prov
        return (f"{len(bad)} incomplete provenances refused by write (no file left); the three roles "
                "round-trip; the loader still takes a file without provenance")

    def no_shared_default():
        # A dataclass instance as a default argument is one object shared
        # by every call; the default must be None and the RingsParams
        # built inside, so a caller's later mutation changes nothing.
        fn, rp_from = require("columns_params_from_rings", "rings_params_from")
        sig = inspect.signature(fn)
        assert sig.parameters["rings_params"].default is None, sig.parameters["rings_params"].default
        defaults = list(fn.__defaults__ or ()) + list((fn.__kwdefaults__ or {}).values())
        assert not any(isinstance(d, RingsParams) for d in defaults), defaults
        prov = {"dataset_role": "exploratory", "date": "2026-09-23", "code_commit": "8f39e0a"}
        a, b = fn(44.0, provenance=prov), fn(44.0, provenance=prov)
        assert a == b and a is not b
        assert rp_from(a) == RingsParams()
        rp = RingsParams(eps_nm=30.0)
        c = fn(44.0, rp, provenance=prov)
        rp.eps_nm = 99.0
        assert c.eps_nm == 30.0, c.eps_nm
        return "rings_params default None; two calls equal, distinct objects; a mutated argument changes nothing"

    def yaml_lazy():
        # Importing tools.mps_columns (the H1 build_rings path) must not
        # need PyYAML: it is imported inside the functions that read or
        # write the parameters file, as mps_metadata and mps_registration
        # do. Checked in a fresh interpreter, and on the module's AST.
        import tools.mps_columns as mc
        code = ("import os, sys; os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen'); "
                "sys.path.insert(0, sys.argv[1]); import tools.mps_columns; print('yaml' in sys.modules)")
        proc = subprocess.run([sys.executable, "-c", code, REPO_ROOT], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert proc.stdout.strip() == "False", proc.stdout
        with open(mc.__file__, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        top = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
        top += [node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)]
        assert not any(name.split(".")[0] == "yaml" for name in top), top
        return "fresh interpreter: 'yaml' not in sys.modules after import; no module-level yaml import"

    def helper_docstrings():
        # Repository convention: every function states its reasoning.
        # _as_float predates H2 (H1, undocumented; pending, not touched).
        import tools.mps_columns as mc
        with open(mc.__file__, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        h2_helpers = ("_param_float", "_param_int", "_param_float_list", "_coerce_columns_value",
                      "_check_columns_ranges", "_check_provenance", "_yaml_module", "_yaml_load_strict")
        for name in h2_helpers:
            assert name in funcs, f"{name} missing"
            assert ast.get_docstring(funcs[name]), f"{name} has no docstring"
        undocumented = [name for name, node in funcs.items()
                        if ast.get_docstring(node) is None and name != "_as_float"]
        assert not undocumented, undocumented
        return f"{len(h2_helpers)} H2 helpers documented; no module-level function without a docstring but _as_float (H1)"

    try:
        check("write -> load round-trip equal on every field (floats, lists, None, provenance dict)", round_trip)
        check("types preserved: floats stay float, ints int, None round-trips", types_preserved)
        check("YAML human-readable: header comment block, keys in dataclass order, safe_load, no tags",
              human_readable)
        check("unknown key -> ValueError naming it; missing tau0_nm -> ValueError; minimal file takes defaults",
              strictness)
        check("repeated key (top level, provenance block or flow) -> ValueError naming it; repeated value loads",
              duplicate_keys)
        check("out-of-range values -> ValueError naming the field on load, write and columns_params_from_rings; "
              "boundary values accepted", ranges)
        check("write refuses a provenance without dataset_role/date/code_commit or with an unknown role "
              "(no file left); the three roles round-trip", provenance_required)
        check("columns_params_from_rings: no RingsParams instance as a default argument", no_shared_default)
        check("PyYAML imported lazily: not in sys.modules after importing tools.mps_columns", yaml_lazy)
        check("H2 helpers have docstrings (repository convention)", helper_docstrings)
        check("rings_params_from copies every shared field exactly, leaves the others at RingsParams defaults",
              rings_params)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================ 6. H1 harness
def test_h1_harness() -> None:
    print("\n6. validate_rings_h1.py in a subprocess (--with-h1)")

    def h1_passes():
        proc = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "validate_rings_h1.py")],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600)
        tail = [ln for ln in proc.stdout.splitlines() if re.search(r"\d+ passed, \d+ failed", ln)]
        assert tail, proc.stdout[-2000:] + proc.stderr[-2000:]
        m = re.search(r"(\d+) passed, (\d+) failed", tail[-1])
        assert m is not None
        passed, failed = int(m.group(1)), int(m.group(2))
        assert (passed, failed) == (70, 0), tail[-1]
        return tail[-1]

    check("validate_rings_h1.py: 70 passed, 0 failed (unchanged counts)", h1_passes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--with-h1", action="store_true",
                        help="also run validate_rings_h1.py in a subprocess (~10 min)")
    args = parser.parse_args()
    print("=" * 72)
    print("COLUMNS H2 CHECKS: bootstrap sigma, suspect marks, tau_0, ColumnsParams")
    print("=" * 72)
    test_truth()
    test_bootstrap()
    small_st = test_small_baseline()
    marks_st = test_marks()
    test_tau(small_st, marks_st)
    test_columns_params()
    if args.with_h1:
        test_h1_harness()
    else:
        print("\n6. validate_rings_h1.py not re-run (pass --with-h1; ~10 min)")
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
