# Use the official Node.js 21 image as the base image
FROM node:21-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy the package.json and package-lock.json (or yarn.lock) files
COPY package.json package-lock.json ./

# Install the dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the Next.js application
RUN npm run build

# Expose the port the application will run on
EXPOSE 3000

# Define the command to run the application
CMD ["npm", "start"]