# Cancer Web App - Docker Deployment

Simple guide to deploy the Cancer Web App using Docker with separate database and frontend containers.

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)

## Quick Deployment

Follow these steps in order:

### 1. Start Database

```bash
docker-compose -f docker-compose.db.yml up -d
```

Wait for database to be ready:

```bash
docker-compose -f docker-compose.db.yml logs postgres
```

Look for: `database system is ready to accept connections`

### 2. Run Database Migration

```bash
npm run db:migrate
```

This creates the necessary database tables and schema.

### 3. Start Frontend Application

```bash
docker-compose up -d
```

### 4. Access Application

- Frontend: http://localhost:3000
- Database: localhost:5432

## Environment Variables

Update these in `docker-compose.yml` before deployment:

```env
DATABASE_URL=postgresql://myuser:mypassword@host.docker.internal:5432/cancer-database
NEXTAUTH_SECRET=your-secret-key-here-change-this-in-production
NEXTAUTH_URL=http://localhost:3000
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
ML_API_URL=https://your-flask-api.com/predict
```

## Stop Services

```bash
# Stop frontend
docker-compose down

# Stop database
docker-compose -f docker-compose.db.yml down
```

## Troubleshooting

1. **Can't access http://localhost:3000**: Check if port 3000 is already in use
2. **Database connection errors**: Ensure database is fully started before running migration
3. **Migration fails**: Make sure `DATABASE_URL` is correctly set in your environment

## Check Logs

```bash
# Frontend logs
docker-compose logs frontend

# Database logs
docker-compose -f docker-compose.db.yml logs postgres
```
