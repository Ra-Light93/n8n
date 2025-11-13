#!/bin/bash
echo "Running git add ."
git add .

echo "Running git commit"
git commit -am "$(date)"

echo "Running git push"
git push

echo "Push is done!"
