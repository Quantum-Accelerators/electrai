#!/bin/bash
aws s3 cp --no-sign-request --recursive s3://materialsproject-parsed/elfcars elfcars
rm elfcars/.*
