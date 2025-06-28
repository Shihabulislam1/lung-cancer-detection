# 

version: '3.8'

services:
  postgres:
    container_name: cancer-database
    image: postgres:15
    restart: always
    environment:
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: mypassword
      POSTGRES_DB: cancer-database
      POSTGRES_INITDB_ARGS: "--locale=en_US.UTF-8 --encoding=UTF8"
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    command: postgres -c 'cluster_name=cancer-database'

volumes:
  postgres_data: