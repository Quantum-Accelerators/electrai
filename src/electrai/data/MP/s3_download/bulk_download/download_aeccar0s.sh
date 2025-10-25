#!/bin/bash
aws s3 cp --no-sign-request --recursive s3://materialsproject-parsed/aeccar0s aeccar0s
rm aeccar0s/.*