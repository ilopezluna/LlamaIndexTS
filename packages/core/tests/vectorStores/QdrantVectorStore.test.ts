import type { BaseNode } from "llamaindex/Node";
import { TextNode } from "llamaindex/Node";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { QdrantClient } from "@qdrant/js-client-rest";
import {
  QdrantVectorStore,
  VectorStoreQueryMode,
} from "llamaindex/storage/index";
import { GenericContainer, type StartedTestContainer } from "testcontainers";

describe("QdrantVectorStore", () => {
  let store: QdrantVectorStore;
  let client: QdrantClient;

  let container: StartedTestContainer;

  beforeAll(async () => {
    container = await new GenericContainer("qdrant/qdrant:v1.7.4")
      .withExposedPorts(6333)
      .start();

    const port = container.getMappedPort(6333);
    const url = `http://${container.getHost()}:${port}`;
    client = new QdrantClient({ url, port });
  });

  beforeEach(async () => {
    const randomName = (Math.random() + 1).toString(36).substring(7);
    await client.createCollection(randomName, {
      vectors: {
        size: 2,
        distance: "Cosine",
      },
    });
    store = new QdrantVectorStore({
      client,
      collectionName: randomName,
      batchSize: 100,
    });
  });

  afterAll(async () => {
    await container.stop();
  });

  describe("[QdrantVectorStore] createCollection", () => {
    it("should create a new collection", async () => {
      await store.createCollection("testCollection", 128);
      expect(await store.collectionExists("testCollection")).equals(true);
    });

    describe("[QdrantVectorStore] add", () => {
      it("should add nodes to the vector store", async () => {
        const nodes: BaseNode[] = [
          new TextNode({
            embedding: [0.1, 0.2],
            metadata: { meta1: "Some metadata" },
          }),
        ];
        const ids = await store.add(nodes);
        expect(ids.length).equals(1);
      });
    });

    describe("[QdrantVectorStore] delete", () => {
      it("should delete from the vector store", async () => {
        const nodes: BaseNode[] = [
          new TextNode({
            embedding: [0.1, 0.2],
            metadata: { meta1: "Some metadata" },
          }),
        ];
        const ids = await store.add(nodes);
        const notEmpty = await client.count(store.collectionName);
        expect(notEmpty.count).equals(1);

        await store.delete(ids[0]);
        const empty = await client.count(store.collectionName);
        expect(empty.count).equals(0);
      });
    });

    describe("[QdrantVectorStore] search", () => {
      it("should search in the vector store", async () => {
        const emptyResult = await store.query({
          queryEmbedding: [0.1, 0.2],
          similarityTopK: 1,
          mode: VectorStoreQueryMode.DEFAULT,
        });
        expect(emptyResult.ids.length).equals(0);

        const nodes: BaseNode[] = [
          new TextNode({
            embedding: [0.1, 0.2],
            metadata: { meta1: "Some metadata" },
          }),
        ];
        await store.add(nodes);

        const searchResult = await store.query({
          queryEmbedding: [0.1, 0.2],
          similarityTopK: 1,
          mode: VectorStoreQueryMode.DEFAULT,
        });
        expect(searchResult.ids.length).equals(1);
      });
    });
  });
});
