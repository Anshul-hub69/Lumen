import json, os, tempfile, threading, unittest, urllib.request
import numpy as np, pandas as pd

from lumen import data, metrics, serve as serve_mod
from lumen.config import Config
from lumen.pipeline import TwoStageRecommender
from lumen.split import build_cases, cutoffs

CFG = Config(n_users=500, n_items=300)
ITEMS, INTER = data.generate(CFG.n_users, CFG.n_items, CFG.horizon, CFG.seed)


class MetricTests(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(metrics.recall_at_k([1, 2, 3], {2, 9}, 3), 0.5)
        self.assertEqual(metrics.hit_at_k([1, 2, 3], {9}, 3), 0.0)
        self.assertAlmostEqual(metrics.ndcg_at_k([7, 1], {7}, 2), 1.0)          # perfect
        self.assertAlmostEqual(metrics.ndcg_at_k([1, 7], {7}, 2), 1 / np.log2(3))


class DataTests(unittest.TestCase):
    def test_items_never_used_before_release(self):
        born = ITEMS.set_index("item_id").born_day
        self.assertTrue((INTER.ts >= INTER.item_id.map(born)).all())

    def test_no_duplicate_pairs(self):
        self.assertFalse(INTER.duplicated(["user_id", "item_id"]).any())

    def test_movielens_loader(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"userId": [1, 1, 2, 2], "movieId": [10, 20, 10, 20], "rating": [5, 3, 4, 4.5],
                          "timestamp": [0, 86400, 172800, 259200]}).to_csv(f"{d}/ratings.csv", index=False)
            pd.DataFrame({"movieId": [10, 20], "title": ["A (2000)", "B (2001)"],
                          "genres": ["Action|Sci-Fi", "Drama"]}).to_csv(f"{d}/movies.csv", index=False)
            items, inter = data.load_movielens(d)
            self.assertEqual(len(inter), 3)                    # the 3-star rating is dropped
            self.assertEqual(list(items.category), ["Action", "Drama"])
            self.assertIn("sci-fi".replace("-", ""), items.tags[0])


class UserSplitTests(unittest.TestCase):
    def test_bursty_data_gets_warm_and_cold_users(self):
        rng = np.random.default_rng(0)
        start = rng.integers(0, 8000, INTER.user_id.max() + 1)
        bursty = INTER.copy()
        bursty["ts"] = start[bursty.user_id] + (bursty.ts % 6)         # every user rates within a 6-day burst
        items2, inter2 = data.to_relative_time(ITEMS, bursty)
        lo, hi, _ = cutoffs(inter2, CFG)
        kinds = [c.kind for c in build_cases(inter2, hi, None, CFG.ctx_len)]
        self.assertGreater(kinds.count("warm"), 100)
        self.assertGreater(kinds.count("cold"), 10)
        # within a user, order is preserved and nothing is lost
        self.assertEqual(len(inter2), len(bursty))
        for _, g in inter2.sort_values(["ts"], kind="stable").groupby("user_id"):
            self.assertTrue(g.ts.is_monotonic_increasing)


class SplitTests(unittest.TestCase):
    def test_no_leakage(self):
        lo, hi, _ = cutoffs(INTER, CFG)
        cases = build_cases(INTER, hi, None, CFG.ctx_len)
        self.assertTrue(cases)
        for c in cases:
            self.assertFalse(set(c.history) & set(c.targets))
            if c.kind == "cold":
                self.assertEqual(len(c.history), CFG.ctx_len)
                past = INTER[(INTER.user_id == c.user) & (INTER.ts < hi)]
                self.assertEqual(len(past), 0)                 # cold users truly have no past


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = TwoStageRecommender(ITEMS, CFG).fit(INTER, max_users=400, verbose=False)

    def test_recommend_contract(self):
        out = self.rec.recommend([1, 2, 3], k=10)
        ids = [o["item_id"] for o in out]
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(set(ids)), 10)
        self.assertFalse(set(ids) & {1, 2, 3})                 # never re-recommend history
        self.assertTrue(all(o["reason"] for o in out))

    def test_cold_start_paths(self):
        self.assertEqual(len(self.rec.recommend([], k=5)), 5)                       # no signal at all
        sci = self.rec.recommend([], prefs=["Sci-Fi"], k=8)                         # onboarding picks
        share = np.mean([o["category"] == "Sci-Fi" for o in sci])
        self.assertGreater(share, 0.3)

    def test_new_items_are_reachable(self):
        hi = cutoffs(INTER, CFG)[1]
        new_ids = set(ITEMS.item_id[ITEMS.born_day >= hi])
        pool = set()
        for h in ([1, 2], [50, 60, 70], [100, 110]):
            pool |= set(self.rec.candidates(h))
        self.assertTrue(pool & new_ids)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.rec.save(f"{d}/m.pkl")
            again = TwoStageRecommender.load(f"{d}/m.pkl")
            self.assertEqual(self.rec.rank([4, 5, 6], mode="ranker"), again.rank([4, 5, 6], mode="ranker"))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        serve_mod.REC = PipelineTests.rec if hasattr(PipelineTests, "rec") else \
            TwoStageRecommender(ITEMS, CFG).fit(INTER, max_users=400, verbose=False)
        from http.server import ThreadingHTTPServer
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_mod.Handler)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def post(self, body):
        req = urllib.request.Request(self.base + "/recommend", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_endpoints(self):
        self.assertEqual(json.loads(urllib.request.urlopen(self.base + "/health").read())["status"], "ok")
        code, out = self.post({"history": [], "categories": ["Drama"], "k": 4})
        self.assertEqual((code, out["mode"], len(out["items"])), (200, "cold_start", 4))
        code, out = self.post({"history": [1, 2, 3, 4], "k": 6})
        self.assertEqual((code, out["mode"]), (200, "personalized"))
        self.assertEqual(self.post({"history": [99999]})[0], 400)
        self.assertEqual(self.post({"history": ["x"]})[0], 400)


if __name__ == "__main__":
    unittest.main()
