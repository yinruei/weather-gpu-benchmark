"""Equal workloads, counterbalanced schedules and CPU/CUDA trace accounting."""
import unittest

from physicsnemo_schedule import CASES, execution_plan, sequence_labels, summarize_trace


class ScheduleTest(unittest.TestCase):
    def test_paired_schedules_have_equal_work_and_counterbalanced_firsts(self):
        plan = execution_plan(12, 8, 42)
        self.assertEqual(plan, execution_plan(12, 8, 42))
        self.assertNotEqual(plan, execution_plan(12, 8, 43))
        for repeat in range(12):
            pair = [b for b in plan if b['round'] == repeat]
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]['first_case'], pair[1]['first_case'])
            for block in pair:
                self.assertEqual([block['sequence'].count(c['case']) for c in CASES], [8, 8])
                labels = sequence_labels(block['sequence'])
                transitions = sum(r['switched'] for r in labels)
                self.assertLessEqual(transitions, 2) if block['schedule'] == 'grouped' else self.assertGreaterEqual(transitions, 15)
        first = [b for b in plan if b['block_order'] == 0]
        self.assertEqual(sum(b['schedule'] == 'grouped' for b in first), 6)
        self.assertEqual(sum(b['first_case'] == CASES[0]['case'] for b in first), 6)
        for case in CASES:
            for schedule in ['grouped', 'alternating']:
                self.assertEqual(sum(b['first_case'] == case['case'] and b['schedule'] == schedule for b in first), 3)
        for args in [(0, 8, 1), (2, 1, 1), (2, 8, -1)]:
            with self.assertRaises(ValueError):
                execution_plan(*args)

    def test_trace_unions_overlap_and_exact_output_bytes(self):
        def event(category, name, start, duration, **extra):
            return dict(ph='X', cat=category, name=name, ts=start, dur=duration, **extra)
        doc = {'traceEvents': [event('user_annotation', 'shape_call/grouped/0/0/b16_s64', 0, 1000),
               event('user_annotation', 'shape_dispatch', 0, 300), event('user_annotation', 'shape_return', 300, 700),
               event('kernel', 'fft_a', 50, 200), event('kernel', 'fft_b', 150, 200),
               event('cuda_runtime', 'cudaLaunchKernel', 10, 50),
               event('cuda_runtime', 'cudaStreamSynchronize', 400, 500),
               event('gpu_memcpy', 'Memcpy DtoH', 800, 50, args={'bytes': 262144})]}
        row = summarize_trace(doc)[0]
        self.assertAlmostEqual(row['kernel_union_ms'], .3)
        self.assertAlmostEqual(row['active_union_ms'], .35)
        self.assertAlmostEqual(row['runtime_wait_ms'], .5)
        self.assertEqual(row['dtoh_bytes'], 262144)
        self.assertEqual(row['dtoh_copies'], 1)
        with self.assertRaises(ValueError):
            summarize_trace({'traceEvents': []})


if __name__ == '__main__':
    unittest.main()
