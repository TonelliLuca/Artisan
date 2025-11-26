
plugins {
    id("java")
    application
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("dev.langchain4j:langchain4j-agentic:1.7.1-beta14")
    implementation("dev.langchain4j:langchain4j-mcp-docker:1.7.1-beta14")
    implementation("dev.langchain4j:langchain4j-mcp:1.7.1-beta14")
    implementation("dev.langchain4j:langchain4j:1.7.1")
    implementation("dev.langchain4j:langchain4j-http-client:1.7.1")
    implementation("dev.langchain4j:langchain4j-ollama:1.7.1")
    implementation("dev.langchain4j:langchain4j-open-ai:1.7.1")
    implementation("dev.langchain4j:langchain4j-embeddings-all-minilm-l6-v2:1.8.0-beta15")
    implementation("org.slf4j:slf4j-api:2.0.17")
    runtimeOnly("org.slf4j:slf4j-simple:2.0.17")
    implementation("ch.qos.logback:logback-classic:1.5.8")

    // Test
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}


tasks.test {
    useJUnitPlatform()
}