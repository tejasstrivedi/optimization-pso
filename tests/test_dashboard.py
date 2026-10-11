from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app


class ReportTests(unittest.TestCase):
    def setUp(self):
        consumers, dt = app.load_workbook_bytes((ROOT / app.DEFAULT_WORKBOOK).read_bytes())
        self.model = app.build_model(consumers, dt)
        self.cfg = app.EvalConfig()
        self.refs = app.baseline_refs(self.model, self.cfg)

    def test_recommendation_and_growth(self):
        ev = app.evaluate_batch(app.report_assignment(self.model), self.model, self.cfg, self.refs)
        self.assertTrue(ev['feasible'][0])
        self.assertAlmostEqual(ev['j'][0], 0.3335541699461108, places=12)
        self.assertEqual((ev['n_t'][0], ev['n_r'][0], ev['distance'][0]), (8, 8, 610))
        self.assertAlmostEqual(np.max(ev['phase_load_pct'][0]) * 1.05, 102.05425166, places=7)

    def test_saved_runs_and_local_refinement(self):
        self.assertFalse(np.array_equal(app.seed_assignment(self.model), app.report_assignment(self.model)))
        for seed, w0, w1, expected, iterations in [
            (1004, .9, .4, .3691700423952783, 64),
            (101, .7, .7, .3639342845396221, 45),
        ]:
            result = app.run_pso(self.model, self.cfg, self.refs, 60, 200, seed,
                                 w0, w1, 2, 2, .2, 30, False, 5, 0, False)
            self.assertAlmostEqual(result['eval']['j'][0], expected, places=12)
            self.assertEqual(result['iterations'], iterations)
            if seed == 1004:
                saved = pd.read_csv(app.REPORT_DIR / 'history_12_1004.csv')
                np.testing.assert_allclose(result['history']['Best_J'], saved['Best_J'], rtol=0, atol=1e-12)
            if seed == 101:
                assignment, ev, moves = app.local_search(result['assignment'], self.model, self.cfg, self.refs)
                self.assertEqual(moves, 9)
                np.testing.assert_array_equal(assignment, app.report_assignment(self.model))

    def test_report_table_9_fixed_plan_scenarios(self):
        # Values transcribed from Table 9 of the final report, at its printed precision.
        for peak, solar, day, scenario, dt_max, phase_max, feasible in [
            (0, 0, 0, 0, 92.68, 97.19, True),
            (.05, 0, 0, 0, 97.31, 102.05, False),
            (0, .2, 0, 1, 54.68, 57.51, True),
            (0, .4, 0, 1, 53.20, 57.51, True),
            (0, .2, .1, 1, 61.05, 63.26, True),
        ]:
            with self.subTest(peak=peak, solar=solar, day=day):
                changed = self.model.consumers.copy()
                changed['Peak_Import_P_kW'] *= 1 + peak
                changed['Peak_Import_Q_kVAr'] *= 1 + peak
                changed['Net_Day_P_kW'] = changed['Daytime_Load_kW'] * (1 + day) - changed['PV_Output_kW'] * (1 + solar)
                changed['Day_Q_kVAr'] *= 1 + day
                model = app.build_model(changed, self.model.dt_master)
                ev = app.evaluate_batch(app.report_assignment(model), model, self.cfg, self.refs)
                self.assertAlmostEqual(ev['load_pct'][0, scenario].max(), dt_max, delta=.005)
                self.assertAlmostEqual(ev['phase_load_pct'][0, scenario].max(), phase_max, delta=.005)
                self.assertEqual(bool(ev['feasible'][0]), feasible)

    def test_dataset_identity(self):
        self.assertTrue(app.matches_report_dataset(self.model))
        self.model.consumers.loc[0, 'Consumer_ID'] = 'CUSTOM'
        self.assertFalse(app.matches_report_dataset(self.model))

    def test_transfer_cap_reverts_longest(self):
        recommended = app.report_assignment(self.model)
        ids = self.model.consumers['Consumer_ID'].tolist()
        assignment = np.zeros_like(recommended)
        for cid in ('C0054', 'C0005', 'C0091'):
            i = ids.index(cid)
            assignment[i] = recommended[i]
        repaired = app.repair_assignment(assignment, self.model, app.EvalConfig(max_transfers=2))
        self.assertEqual(repaired[ids.index('C0054')], 0)
        self.assertNotEqual(repaired[ids.index('C0005')], 0)
        self.assertNotEqual(repaired[ids.index('C0091')], 0)


class StreamlitTests(unittest.TestCase):
    def test_animation_includes_early_stop_in_single_chart(self):
        import pyarrow as pa
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
        self.assertFalse(next(c for c in at.checkbox if c.label == 'Apply local-search refinement after PSO').value)
        next(b for b in at.button if b.label == 'Run PSO').click().run()
        self.assertFalse(at.exception)
        charts = at.get('vega_lite_chart')
        self.assertEqual(len(charts), 1)
        plotted = pa.ipc.open_stream(charts[0].proto.datasets[0].data.data).read_all().to_pandas()
        self.assertEqual(plotted.columns.tolist(), ['Iteration', 'Best_J'])
        self.assertEqual(int(plotted.iloc[-1]['Iteration']), 64)
        self.assertAlmostEqual(plotted.iloc[-1]['Best_J'], .3691700423952783, places=12)

    def test_render_and_live_run(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.tabs), 6)
        target = next(s for s in at.sidebar.slider if s.label == 'Preferred DT loading target (%)')
        target.set_value(85).run()
        self.assertFalse(at.exception)
        self.assertTrue(any('reproduces the revised' in s.value for s in at.success))
        next(s for s in at.sidebar.slider if s.label == 'Preferred DT loading target (%)').set_value(90).run()
        for checkbox in at.checkbox:
            if checkbox.label in ('Animate during the run', 'Apply local-search refinement after PSO'):
                checkbox.uncheck()
        at.run()
        next(b for b in at.button if b.label == 'Run PSO').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(int(at.session_state['last_history'].iloc[-1]['Iteration']), 64)
        at.sidebar.selectbox[0].select('Recommendation origin: run 101').run()
        next(c for c in at.checkbox if c.label == 'Apply local-search refinement after PSO').check().run()
        next(b for b in at.button if b.label == 'Run PSO').click().run()
        self.assertFalse(at.exception)
        consumers, dt = app.load_workbook_bytes((ROOT / app.DEFAULT_WORKBOOK).read_bytes())
        np.testing.assert_array_equal(at.session_state['last_assignment'], app.report_assignment(app.build_model(consumers, dt)))


if __name__ == '__main__':
    unittest.main()
