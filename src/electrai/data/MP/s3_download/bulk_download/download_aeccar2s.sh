#!/bin/bash
aws s3 cp --no-sign-request --recursive s3://materialsproject-parsed/aeccar2s aeccar2s
rm aeccar2s/.*