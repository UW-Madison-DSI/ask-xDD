# Load .env into memory as environment variable

export $(grep -v '^#' .env | xargs -d '\n')