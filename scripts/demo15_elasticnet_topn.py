#!/usr/bin/env python3
"""
Runs the ElasticNet Top-N strategy on the DEMO15 universe.
"""

from mlbt.pipelines.wrappers import run_demo_elasticnet_topn


def main():
    run_demo_elasticnet_topn()



if __name__ == "__main__":
    main()