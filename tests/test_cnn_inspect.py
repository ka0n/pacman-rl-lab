# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only loss report and replay selection checks."""
import json
from pathlib import Path
import numpy as np
import pytest
from pacman_lab.cnn_inspect import _death_type, audit
from pacman_lab.cnn_batch import eval_batched
from pacman_lab.cnn_models import make_model
from pacman_lab.marathon_train import atomic_save_torch


def test_death_classifier():
    before = np.zeros((4,7,9), dtype='uint8')
    after = before.copy()
    before[2,1,1]=1
    before[3,1,2]=1
    after[2,1,2]=1
    assert _death_type(before,after,False) == 'player_entered_ghost'
    assert _death_type(before,after,True) == 'timeout'
    after[2,1,2]=0
    after[2,2,1]=1
    assert _death_type(before,after,False) == 'ghost_moved_or_swapped'


def test_audit_same_policy_as_v09(tmp_path: Path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    model = make_model('cnn_single', (4,7,9))
    cp = tmp_path/'best.pt'
    atomic_save_torch({'architecture':'cnn_single','input_shape':(4,7,9),
                      'model':model.state_dict(),'config':{'maze':'small','seed':42},
                       'step':12}, cp)
    out = tmp_path/'loss_report.json'
    report = audit(cp,maze='small',seed=70000,episodes=8,worlds=3,
                   engine='python',device_name='cpu',output=out)
    assert out.is_file()
    reference = eval_batched(model,'cnn_single',torch.device('cpu'),
                             maze='small',episodes=8,seed=70000,
                             worlds=3,engine='python')
    assert report['summary']['wins'] == reference['wins']
    assert abs(report['summary']['pellet_percent']-reference['pellet_percent']) < 0.011
    assert len(report['episodes_detail']) == 8
    assert {e['seed'] for e in report['episodes_detail']} == set(range(70000,70008))
    assert report['summary']['losses'] == sum(report['summary']['loss_reasons'].values())
    assert all(e['reason'] in ('win','timeout','player_entered_ghost','ghost_moved_or_swapped')
               for e in report['episodes_detail'])
    with pytest.raises(FileExistsError):
        audit(cp,maze='small',seed=70000,episodes=2,worlds=2,
              engine='python',device_name='cpu',output=out)
