import argparse
import logging
from drivers import main
import streamlit as st

logging.basicConfig(level=logging.ERROR) 

if __name__ == '__main__':
    main()