import torch
import torch.nn as nn 

class interpolation_layer(nn.Module):
    def __init__(self):
        super(interpolation_layer, self).__init__()

    def forward(self, feature_maps, init_potential_anchor):
        """
        :param feature_map: (Bs, 256, Height, Width)
        :param potential_anchor: (BS, number_point, 2)
        :return:
        """

        feature_dim = feature_maps.size()
        
        potential_anchor = init_potential_anchor * (feature_dim[2] - 1)

        potential_anchor = torch.clamp(potential_anchor, 0, feature_dim[2] - 1)

        anchor_pixel = self._get_interploate(potential_anchor, feature_maps, feature_dim)
        return anchor_pixel


    def _flatten_tensor(self, input):
        return input.contiguous().view(input.nelement())


    def _get_index_point(self, input, anchor, feature_dim):
        point_shape = anchor.size()

        index = anchor[:, :, 1] * feature_dim[2] + anchor[:, :, 0]

        batch_index = (torch.arange(0, feature_dim[0], dtype=index.dtype, device=index.device) * (
                    feature_dim[2] * feature_dim[3])).unsqueeze(1)
        index = (index + batch_index).flatten(0)

        output = torch.index_select(input.permute(1, 0, 2, 3).contiguous().flatten(1), 1, index)
        output = output.view(feature_dim[1], feature_dim[0], point_shape[1])

        return output.permute(1, 2, 0).contiguous()


    def _get_interploate(self, potential_anchor, feature_maps, feature_dim):
        anchors_lt = potential_anchor.floor().long()
        anchors_rb = potential_anchor.ceil().long()

        anchors_lb = torch.stack([anchors_lt[:, :, 0], anchors_rb[:, :, 1]], 2)
        anchors_rt = torch.stack([anchors_rb[:, :, 0], anchors_lt[:, :, 1]], 2)

        vals_lt = self._get_index_point(feature_maps, anchors_lt.detach(), feature_dim)
        vals_rb = self._get_index_point(feature_maps, anchors_rb.detach(), feature_dim)
        vals_lb = self._get_index_point(feature_maps, anchors_lb.detach(), feature_dim)
        vals_rt = self._get_index_point(feature_maps, anchors_rt.detach(), feature_dim)

        coords_offset_lt = potential_anchor - anchors_lt.type(potential_anchor.data.type())

        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, :, 0:1]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, :, 0:1]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, :, 1:2]

        return mapped_vals



class get_roi(nn.Module):

    def __init__(self, num_points, half_length, img_size):
        super(get_roi, self).__init__()
        self.img_size = img_size
        self.num_points = num_points
        self.half_length = torch.tensor([[[half_length, half_length]]], dtype=torch.float32)
        self.half_length.requires_grad = False

    def forward(self, anchor):
        Bs = anchor.size(0)
        half_length = (self.half_length.to(anchor.device) / (self.img_size)).repeat(Bs, 1, 1)
        bounding_min = torch.clamp(anchor - half_length, 0.0, 1.0)
        bounding_max = torch.clamp(anchor + half_length, 0.0, 1.0)
        bounding_box = torch.cat((bounding_min, bounding_max), dim=2)
        bounding_length = bounding_max - bounding_min

        bounding_xs = torch.nn.functional.interpolate(bounding_box[:,:,0::2], size=self.num_points,
                                                      mode='linear', align_corners=True)
        bounding_ys = torch.nn.functional.interpolate(bounding_box[:,:,1::2], size=self.num_points,
                                                      mode='linear', align_corners=True)
        bounding_xs, bounding_ys = bounding_xs.unsqueeze(3).repeat_interleave(self.num_points, dim=3), \
                                   bounding_ys.unsqueeze(2).repeat_interleave(self.num_points, dim=2)

        meshgrid = torch.stack([bounding_xs, bounding_ys], dim=-1)

        return meshgrid, bounding_length, bounding_min

