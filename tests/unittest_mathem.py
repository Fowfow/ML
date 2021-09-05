#!/usr/bin/python3

import unittest
import sys
import os

parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#sys.path.append(parent_directory) 
sys.path.append("../")

from tools.mathem import basic




class TestMathem(unittest.TestCase):
    def test_factorial(self):
        r   = [basic.factorial(n) for n in range(6)]
        exp = [1, 1, 2, 6, 24, 120]
        self.assertEqual(r, exp)



        
         
# from tools.mathem import basic
if __name__ == "__main__":
    unittest.main()