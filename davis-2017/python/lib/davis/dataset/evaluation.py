# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# ----------------------------------------------------------------------------

from collections import defaultdict

import itertools
import numpy as np
import skimage.morphology
from easydict import EasyDict as edict
from prettytable import PrettyTable

from .. import measures
from ..misc import log
from ..misc.config import cfg
from ..misc.parallel import Parallel, delayed

_db_measures = {
        'J': measures.db_eval_iou,
        'F': measures.db_eval_boundary,
        'T': measures.db_eval_boundary,
        }

def db_eval_sequence(segmentations,annotations,measure='J',n_jobs=cfg.N_JOBS):

  """
  Evaluate video sequence results.

	Arguments:
		segmentations (list of ndarrya): segmentations masks.
		annotations   (list of ndarrya): ground-truth  masks.
    measure       (char): evaluation metric (J,F,T)
    n_jobs        (int) : number of CPU cores.

  Returns:
    results (list): ['raw'] per-frame, per-object sequence results.
  """
  # import pdb; pdb.set_trace()
  log.info("Evaluating measure: {} on {}".format(measure, segmentations.name))
  segmentations_copy = [None]*len(segmentations)
  for i in range(len(segmentations)):
    new_sg = segmentations[i].copy()
    # import pdb
    # pdb.set_trace()
    new_sg[segmentations[i]==255] = 1
    segmentations_copy[i] = new_sg

  results = {'raw':[]}
  for obj_id in annotations.iter_objects_id():
    # import pdb; pdb.set_trace()
    results['raw'].append(Parallel(n_jobs=n_jobs)(delayed(_db_measures[measure])(
      an==obj_id,sg==obj_id) for an,sg in zip(annotations[1:-1],segmentations_copy[1:-1])))

    # test code to confirm that --single-object flag is used to combine all id into 1
    # test2 = [_ for _ in segmentations_copy[1:-1]]
    # import numpy as np
    # np.unique(test2[0])
    # test1 = [_ for _ in annotations[1:-1]]
    # np.unique(test1[0])
    # np.sum(test1[0]==0)
    # np.sum(test2[0] == 0)

  for stat,stat_fuc in measures._statistics.iteritems():
    results[stat] = [float(stat_fuc(r)) for r in results['raw']]

  # Convert to 'float' to save correctly in yaml format
  for r in range(len(results['raw'])):
    results['raw'][r] = [float(v) for v in results['raw'][r]]

  results['raw'] = [[np.nan]+r+[np.nan] for r in results['raw']]

  return results

def db_eval(db, segmentations, measures, n_jobs=cfg.N_JOBS, verbose=True):

  """
  Evaluate video sequence results.

	Arguments:
		segmentations (list of ndarrya): segmentations masks.
		annotations   (list of ndarrya): ground-truth  masks.
    measure       (char): evaluation metric (J,F,T)
    n_jobs        (int) : number of CPU cores.

  Returns:
    results (dict): [sequence]: per-frame sequence results.
                    [dataset] : aggreated dataset results.
  """

  s_eval = defaultdict(dict)  # sequence evaluation
  d_eval = defaultdict(dict)  # dataset  evaluation

  for measure in measures:
    log.info("Evaluating measure: {}".format(measure))
    for sid in range(len(db)):
      sg = segmentations[sid]
      s_eval[sg.name][measure] = db_eval_sequence(sg,
          db[sg.name].annotations, measure=measure, n_jobs=n_jobs)

    for statistic in cfg.EVAL.STATISTICS:
      raw_data = np.hstack([s_eval[sequence][measure][statistic] for sequence in
        s_eval.keys()])
      d_eval[measure][statistic] = float(np.mean(raw_data))

  g_eval = {'sequence':dict(s_eval),'dataset':dict(d_eval)}

  return g_eval

def print_results(evaluation,method_name="-"):
  """Print result in a table"""

  metrics = evaluation['dataset'].keys()

  # Print results
  table = PrettyTable(['Method']+[p[0]+'_'+p[1] for p in
    itertools.product(metrics,cfg.EVAL.STATISTICS)])

  table.add_row([method_name]+["%.3f"%np.round(
    evaluation['dataset'][metric][statistic],3) for metric,statistic in
    itertools.product(metrics,cfg.EVAL.STATISTICS)])

  print ("\n{}\n".format(str(table)))
