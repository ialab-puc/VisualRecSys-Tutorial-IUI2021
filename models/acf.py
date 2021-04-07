import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ACFFeatureNet(nn.Module):
    """
    Process auxiliary item features into latent space.
    All items for user can be processed in batch.
    """
    def __init__(self, emb_dim, input_feature_dim, feature_dim, hidden_dim=None, output_dim=None):
        super().__init__()

        if not hidden_dim:
            hidden_dim = emb_dim

        if not output_dim:
            output_dim = emb_dim

        # e.g. 2048 => 128
        self.dim_reductor = nn.Linear(input_feature_dim, feature_dim)

        self.w_x = nn.Linear(feature_dim, hidden_dim)
        self.w_u = nn.Linear(emb_dim, hidden_dim)

        self.w = nn.Linear(hidden_dim, 1)

        self._kaiming_(self.w_x)
        self._kaiming_(self.w_u)
        self._kaiming_(self.w)

    def _kaiming_(self, layer):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        torch.nn.init.zeros_(layer.bias)

    def forward(self, user, components, profile_mask, return_attentions=False):
        x = self.dim_reductor(components) # Add
        x = x.movedim(0, -2) # BxPxHxD => PxHxBxD

        x_tilde = self.w_x(x)
        user = self.w_u(user)

        beta = F.relu(x_tilde + user)
        beta = self.w(beta)

        beta = F.softmax(beta, dim=1)

        x = (beta * x).sum(dim=1)
        x = x.movedim(-2, 0) # PxBxD => BxPxD

        feature_dim = x.shape[-1]
        profile_mask = profile_mask.float()
        profile_mask = profile_mask.unsqueeze(-1).expand((*profile_mask.shape, feature_dim))

        x = profile_mask * x
        output = {'pooled_features': x}
        if return_attentions:
            output['attentions'] = beta.squeeze(-1).squeeze(-1)
        return output


class ACFUserNet(nn.Module):
    """
    Get user embedding accounting to surpassed items
    """

    def __init__(self, users, items, emb_dim=128, input_feature_dim=0, profile_embedding=None, device=None):
        super().__init__()
        self.pad_token = 0

        self.emb_dim = emb_dim
        num_users = max(users) + 1
        num_items = max(items) + 1

        reduced_feature_dim = emb_dim # TODO: parametrize
        self.feats = ACFFeatureNet(emb_dim, input_feature_dim, reduced_feature_dim) if input_feature_dim > 0 else None

        self.user_embedding = nn.Embedding(num_users, emb_dim)
        if not profile_embedding:
            self.profile_embedding = nn.Embedding(num_items, emb_dim, padding_idx=self.pad_token)
        else:
            self.profile_embedding = profile_embedding

        f = 1 if self.feats is not None else 0
        self.w_u = nn.Linear(emb_dim, emb_dim)
        self.w_v = nn.Linear(emb_dim, emb_dim)
        self.w_p = nn.Linear(emb_dim, emb_dim)
        self.w_x = nn.Linear(emb_dim, emb_dim)
        self.w = nn.Linear(emb_dim, 1)

        self._kaiming_(self.w_u)
        self._kaiming_(self.w_v)
        self._kaiming_(self.w_p)
        self._kaiming_(self.w_x)
        self._kaiming_(self.w)

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.device = device

    def _kaiming_(self, layer):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        torch.nn.init.zeros_(layer.bias)

    def forward(self, user_ids, profile_ids, features, profile_mask, return_component_attentions=False,
                return_profile_attentions=False, return_attentions=False):
        return_component_attentions = return_component_attentions or return_attentions
        return_profile_attentions = return_profile_attentions or return_attentions

        batch_size = user_ids.nelement()
        user = self.user_embedding(user_ids)

        if profile_ids.nelement() != 0:
            profile = self.profile_embedding(profile_ids)
        else:
            profile = torch.zeros((batch_size, 0, self.emb_dim), device=self.device)

        if self.feats is not None:
            features = features.flatten(start_dim=2, end_dim=3) # Add
            feat_output = self.feats(user, features, profile_mask, return_attentions=return_component_attentions)
            components = feat_output['pooled_features']
        else:
            components = torch.tensor([], device=self.device)

        user = self.w_u(user)
        profile_attention = self.w_p(profile) # TODO: Better name
        components = self.w_x(components)

        profile_attention = profile_attention.permute((1,0,2))
        components = components.permute((1,0,2))

        alpha = F.relu(user + profile_attention + components) # TODO: + item, Add curent_item emb (?)
        alpha = self.w(alpha)

        profile_mask = profile_mask.permute((1,0))
        profile_mask = profile_mask.unsqueeze(-1)
        alpha = alpha.masked_fill(torch.logical_not(profile_mask), float('-inf'))
        alpha = F.softmax(alpha, dim=0)

        is_nan = torch.isnan(alpha)
        if is_nan.any():
            # softmax is nan when all elements in dim 0 are -infinity or infinity
            alpha = alpha.masked_fill(is_nan, 0.0)

        alpha = alpha.permute((1,0,2))
        user_profile = (alpha * profile).sum(dim=1)

        user = user + user_profile
        output = {'user': user}
        if return_component_attentions:
            output['component_attentions'] = feat_output['attentions']
        if return_profile_attentions:
            output['profile_attentions'] = alpha.squeeze(-1)

        return output

    @property
    def params(self):
        params_to_update = []
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        return params_to_update


class ACF(nn.Module):
    def __init__(self,
                 users,
                 items,
                 feature_path,
                 model_dim=128,
                 input_feature_dim=0,
                 tied_item_embedding=True,
                 device=None):

        super().__init__()
        self.pad_token = 0
        self.device = device

        # Should be moved to an ACFRecommender
        self.users = users
        self.items = items
        self.feature_path = feature_path
        self.model_dim = model_dim
        self.input_feature_dim = input_feature_dim

        self.all_items = torch.tensor(items)
        self.all_items = self.all_items + 1 if self.all_items.min() == 0 else self.all_items
        self.feature_data = self.load_feature_data(feature_path)
        num_items = max(self.all_items) + 1

        input_feature_dim = self.feature_data.shape[-1]
        self.item_model = nn.Embedding(num_items, self.model_dim, padding_idx=self.pad_token)
        self.user_model = (
            ACFUserNet(
                users,
                items,
                emb_dim=self.model_dim,
                input_feature_dim=input_feature_dim,
                profile_embedding=self.item_model,
                device=self.device)
        if tied_item_embedding else
            ACFUserNet(
                users,
                items,
                emb_dim=self.model_dim,
                input_feature_dim=input_feature_dim,
                device=self.device)
        )

    def forward(self, user_id, profile_ids, pos, neg, profile_mask):
        profile_features = self.get_features(profile_ids).to(self.device)

        user_output = self.user_model(user_id, profile_ids, profile_features, profile_mask)
        user = user_output['user']

        pos_pred = self.get_predictions(user, pos)
        neg_pred = self.get_predictions(user, neg)

        return pos_pred, neg_pred

    def get_predictions(self, user, items):
        item_embeddings = self.item_model(items)
        prediction = self.score(user, item_embeddings)
        return prediction

    def score(self, user, items):
        return (user * items).sum(1) / self.model_dim

    def recommend_all(self, user_id, profile_ids, return_attentions=False):
        # TODO: Improve
        profile_mask = (profile_ids != 0).to(self.device)
        profile_features = self.get_features(profile_ids).to(self.device)

        user_output = self.user_model(user_id, profile_ids, profile_features, profile_mask, return_attentions=return_attentions)
        user = user_output['user']

        all_items = self.all_items.to(self.device)
        item_embeddings = self.item_model(all_items)
        scores = self.score(user, item_embeddings)

        if return_attentions:
            component_attentions = user_output['component_attentions']
            profile_attentions = user_output['profile_attentions']
            return scores, component_attentions, profile_attentions

        return scores

    def load_feature_data(self, feature_path):
        with open(feature_path, 'rb') as fp:
            feature_data = np.load(fp, allow_pickle=True)
        feature_data = feature_data[:,1].tolist()
        feature_data = np.array(feature_data) # Faster when transformed to numpy first
        feature_data = torch.tensor(feature_data)
        feature_data = feature_data.permute((0,2,3,1)) # TODO: Hack: by default d should be last dimension
        feature_data = self.append_default_features(feature_data)
        return feature_data

    def append_default_features(self, feature_data):
        feature_dims = feature_data.shape[1:]
        default_features = torch.zeros((1, *feature_dims))
        feature_data = torch.cat((default_features, feature_data), dim=0)
        return feature_data

    def get_features(self, ids):
        if isinstance(ids, int):
                ids = torch.tensor([ids])
        if isinstance(ids, list):
                ids = torch.tensor(ids)

        return self.feature_data[ids]

    def args(self):
        return {
            'users': self.users,
            'items': self.items,
            'feature_path': self.feature_path,
            'model_dim': self.model_dim,
            'input_feature_dim': self.input_feature_dim,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, device=None):
        args = checkpoint['model_args']
        model = cls(
            users=args['users'],
            items=args['items'],
            feature_path=args['feature_path'],
            model_dim=args['model_dim'],
            input_feature_dim=args['input_feature_dim'],
            device=device,
        )
        model.load_state_dict(checkpoint['state_dict'])
        if device:
            model = model.to(device)

        return model
