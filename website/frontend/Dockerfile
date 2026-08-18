# Frontend build: Vite -> static files served by nginx.
FROM node:20-alpine AS build
WORKDIR /app
COPY website/frontend/package.json ./
RUN npm install
COPY website/frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
