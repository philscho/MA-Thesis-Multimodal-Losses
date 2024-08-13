# CIFAR-10 dataloader -- OG
        # if dataloader_idx == 1:
        #     images, tokens, token_type, mask, label = batch
        #     # Remove batch dimension from text embeddings (if batch_size = 1)
        #     tokens.squeeze_(0), token_type.squeeze_(0), mask.squeeze_(0)
            
        #     outputs = self.model(
        #         pixel_values=images,
        #         input_ids=tokens,
        #         attention_mask=mask,
        #         token_type_ids=token_type
        #     )

        #     image_out = torch.nn.functional.normalize(outputs['image_embeds'], dim=-1)
        #     text_out = torch.nn.functional.normalize(outputs['text_embeds'], dim=-1)
        #     # (b, 1, h_dim) x (b, h_dim, classes) => (b, 1, classes)
        #     #sim_scores = torch.squeeze(torch.bmm(image_out.unsqueeze_(1), text_out.transpose(1, 2)), 1)
        #     sim_scores = image_out @ text_out.transpose(0, 1)
        #     result = sim_scores.argmax(-1) == label
        #     #self.cifar_results.append(result.item())
        #     self.cifar_results.append(result)
        
        # if dataloader_idx == 1:
        #     images, label = batch
            
        #     image_features = self.model.get_image_features(pixel_values=images)
        #     image_out = torch.nn.functional.normalize(image_features, dim=-1)
            
        #     # (b, 1, h_dim) x (b, h_dim, classes) => (b, 1, classes)
        #     #sim_scores = torch.squeeze(torch.bmm(image_out.unsqueeze_(1), text_out.transpose(1, 2)), 1)
        #     sim_scores = image_out @ self.cifar10_classifier
        #     result = sim_scores.argmax(-1) == label
        #     #self.cifar_results.append(result.item())
        #     self.cifar_results.append(result)

# def on_validation_epoch_end(self):
        # #return 
        # preds = torch.stack(self.cifar_results).flatten()
        # #accuracy = sum(self.cifar_results) / len(self.cifar_results)
        # accuracy = preds.sum() / len(preds)
        # self.log("cifar10-accuracy", accuracy, sync_dist=True)
        # self.cifar_results = []