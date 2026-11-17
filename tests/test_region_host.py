"""Actual host state/export workflow, not mocked UI state transitions."""
import json
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from lockframe.region_check import check_regions, run_region_check


def test_real_mainwindow_region_workflow_and_source_time_exports(tmp_path):
    report = check_regions(tmp_path)
    assert report['three_columns'] and report['timeline_full_width']
    assert report['actual_local_tracking']['measured_nonmanual'] == 3
    assert report['single_export']['frames'] == report['batch_export']['frames'] == 4
    assert report['single_export']['audio'] and report['batch_export']['audio']
    assert len(list(tmp_path.glob('三栏区域工作区_*.png'))) == 2


def test_region_check_failure_report_does_not_claim_success(tmp_path, monkeypatch):
    from lockframe import region_check
    def broken(_folder):
        raise RuntimeError('deliberate diagnostic failure')
    monkeypatch.setattr(region_check, 'check_regions', broken)
    assert run_region_check(tmp_path) == 1
    report = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert report['ok'] is False and 'deliberate diagnostic failure' in report['error']
    assert 'regions' not in report
