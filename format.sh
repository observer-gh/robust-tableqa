#!/bin/bash

# Format all Python files
echo "Formatting Python files with autopep8..."
find . -name "*.py" -exec autopep8 --in-place --aggressive --aggressive {} \;



echo "Formatting complete!"