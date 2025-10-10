#!/usr/bin/env python3
"""
Runs the ElasticNet Top-N strategy on the DEMO15 universe.
"""

from mlbt.pipelines.demo15 import run_demo15_elasticnet_topn


def main():
    run_demo15_elasticnet_topn()



if __name__ == "__main__":
    main()