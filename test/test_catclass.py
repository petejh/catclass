import unittest

import catclass

class TruthTest(unittest.TestCase):
    def test_it_knows_the_truth_when_it_sees_it(self):
        self.assertTrue(True)

    def test_it_knows_the_truth_when_it_does_not_see_it(self):
        self.assertTrue(False)
