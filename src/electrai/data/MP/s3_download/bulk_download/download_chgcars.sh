#!/bin/bash
aws s3 cp --no-sign-request --recursive s3://materialsproject-parsed/chgcars chgcars
rm chgcars/.*