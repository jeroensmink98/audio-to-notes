


# for each m4a file in the input folder run 
# uv run --env-file .env main.py <file>

for file in input/*.m4a; do
    uv run --env-file .env main.py "$file"
done