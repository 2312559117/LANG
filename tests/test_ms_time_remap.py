"""Managed-oracle comparisons plus product boundary and duration contracts."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from lockframe.time_remap_model import (
    active_points, default_config, make_constant_speed, normalize_config,
    normalize_keyframes,
)
from lockframe.time_remap_math import (
    basic_speed, convert_basic_to_advanced, evaluate_source, evaluate_sources,
    evaluate_speed,
)
from lockframe.time_remap_map import (
    MAX_OUTPUT_FRAMES, build_frame_map, nearest_source_indices, output_frame_map,
)

@pytest.mark.parametrize('speed,count',[(.5,240),(2.,60),(1.,120),(-.5,240),(-2.,60)])
def test_constant_speed_has_physical_duration_and_correct_direction(speed,count):
    config=make_constant_speed(default_config(),speed)
    frame_map=output_frame_map(config,120)
    assert len(frame_map)==count
    assert abs(frame_map[0]-(0 if speed>0 else 119))<=1e-9
    assert abs(frame_map[-1]-(119 if speed>0 else 0))<=1e-9
    assert np.all(np.diff(frame_map)*speed>=0)


@pytest.mark.parametrize('speed,count',[(.5,240),(2.,60)])
def test_basic_product_mapping_restores_integrated_duration(speed,count):
    config=default_config()
    config.update(enabled=True)
    config['basic']['speeds']=[speed]*5
    # The public legacy conversion faithfully normalizes to 1. The product
    # keeps integral dt=ds/v so a genuine .5x clip has twice as many frames.
    assert convert_basic_to_advanced(config)[-1]['position']==1.
    assert active_points(config)[-1]['position']==pytest.approx(1/speed)
    assert len(output_frame_map(config,120))==count


def test_disabled_preserves_exact_source_frames_even_with_reverse_draft():
    config=make_constant_speed(None,-.5)
    config['enabled']=False
    np.testing.assert_array_equal(output_frame_map(config,107),np.arange(107))


def test_identity_is_exact_and_cached_independent_of_flow_and_audio():
    config=default_config();config['enabled']=True
    first=output_frame_map(config,401)
    np.testing.assert_array_equal(first,np.arange(401))
    other=deepcopy(config)
    other.update(optical_flow=True,preview_flow=True,audio_policy='mapped',flow_backend='dis')
    assert output_frame_map(other,401) is first
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[1]=9
    assert output_frame_map(make_constant_speed(config,.5),401) is not first


def test_reverse_hold_and_local_repeat_are_distinct_from_source_order():
    points=[dict(position=0,source=0,interpolation='linear'),
            dict(position=.4,source=.8,incoming_interpolation='linear',interpolation='linear'),
            dict(position=.7,source=.8,incoming_interpolation='linear',interpolation='linear'),
            dict(position=1,source=.2,incoming_interpolation='linear')]
    values=build_frame_map(101,points)
    assert values[40]==pytest.approx(80)
    assert np.allclose(values[40:71],80)
    assert values[-1]==pytest.approx(20)
    assert np.all(np.diff(values[71:])<0)


def test_rounding_is_away_from_zero_at_positive_half_frame():
    np.testing.assert_array_equal(nearest_source_indices([0,.49,.5,1.5,2.5,9.9]),[0,0,1,2,3,10])


def test_one_source_frame_never_requests_a_nonexistent_neighbor():
    assert build_frame_map(1,[dict(position=0,source=0),dict(position=100,source=1)]).tolist()==[0.]


@pytest.mark.parametrize('count',[0,-1,True,1.,MAX_OUTPUT_FRAMES+1])
def test_invalid_source_count_is_rejected_before_map_allocation(count):
    with pytest.raises(ValueError):
        output_frame_map(default_config(),count)


def test_overlong_map_is_rejected_before_allocation():
    with pytest.raises(ValueError,match='超过'):
        build_frame_map(MAX_OUTPUT_FRAMES,[dict(position=0,source=0),dict(position=100,source=1)])


@pytest.mark.parametrize('values',[[],[np.nan],[np.inf],[-.1],[[0,1]],[MAX_OUTPUT_FRAMES]])
def test_bad_nearest_map_is_rejected(values):
    with pytest.raises(ValueError):
        nearest_source_indices(values)


@pytest.mark.parametrize('raw',[
    {}, {'version':1}, {'version':True}, {'version':2,'enabled':1},
    {'version':2,'mode':'legacy'}, {'version':2,'audio_policy':'copy'},
    {'version':2,'flow_backend':'rife'}, {'version':2,'time_remap':{}},
    {'version':2,'basic':{'speeds':[1]*4}},
    {'version':2,'points':[{'position':float('nan'),'source':0}]},
    {'version':2,'points':[{'position':0,'source':float('inf')}]},
    {'version':2,'points':[{'position':0,'source':0,'outgoing_speed':float('inf')}]},
    {'version':2,'points':[{'position':0,'source':0,'unknown':4}]},
    {'version':2,'points':[{'position':0,'source':0}]*513},
])
def test_versioned_config_rejects_damaged_or_old_data(raw):
    with pytest.raises(ValueError):
        normalize_config(raw)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_evaluation_position_is_rejected(value):
    with pytest.raises(ValueError):evaluate_source(None,value)
    with pytest.raises(ValueError):evaluate_speed(None,value)
    with pytest.raises(ValueError):basic_speed(None,value)


def test_normalizing_and_changing_speed_never_mutates_input():
    config=default_config();before=deepcopy(config)
    normalized=normalize_config(config)
    normalized['basic']['speeds'][0]=9
    make_constant_speed(config,2)
    assert config==before


def test_singleton_and_extreme_points_remain_finite_and_ordered():
    points=normalize_keyframes([dict(position=110,source=2),dict(position=-1,source=-1),dict(position=110,source=.5)])
    assert np.all(np.diff([point['position'] for point in points])>0)
    values=build_frame_map(17,points)
    assert np.isfinite(values).all()
    assert np.min(values)>=0 and np.max(values)<=16


def test_product_single_source_frame_freeze_respects_output_duration():
    config=make_constant_speed(speed=.1)
    config['points'][0]['source']=config['points'][1]['source']=.5
    mapping=output_frame_map(config,1)
    assert len(mapping)==10 and np.array_equal(mapping,np.zeros(10))
    assert not mapping.flags.writeable


def test_extreme_basic_boundaries_reserve_space_for_all_four_sections():
    config=default_config();config['basic']['boundaries']=[100,100,100]
    points=normalize_config(config)['basic']['boundaries']
    assert points==pytest.approx([.8,.88,.96])
    assert all(right-left>=.08-1e-12 for left,right in zip([0,*points],points))
    assert np.isfinite(output_frame_map(config,100)).all()


def test_huge_integer_is_rejected_as_a_readable_validation_error():
    config=default_config();config['points'][0]['source']=10**1000
    with pytest.raises(ValueError,match='有限'):
        normalize_config(config)
    with pytest.raises(ValueError,match='有限'):
        evaluate_source(None,10**1000)
