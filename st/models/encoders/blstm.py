from .encoder import Encoder

from st.models.layers import layer_norm, build_cell, cell_forward


class BLSTM(Encoder):

    def encode(self, features, len_feas):
        # build model in one device
        num_cell_units = self.args.model.encoder.num_cell_units
        num_cell_project = self.args.model.encoder.num_cell_project
        num_layers = self.args.model.encoder.num_layers
        cell_type = self.args.model.encoder.cell_type
        dropout = self.args.model.encoder.dropout
        forget_bias = self.args.model.encoder.forget_bias

        use_residual = self.args.model.use_residual
        use_layernorm = self.args.model.use_layernorm

        hidden_output = features

        for i in range(num_layers):
            # build one layer: build block, connect block
            single_cell = build_cell(
                num_units=num_cell_units,
                num_layers=1,
                is_train=self.is_train,
                cell_type=cell_type,
                dropout=dropout,
                forget_bias=forget_bias,
                use_residual=use_residual,
                dim_project=num_cell_project)
            hidden_output, _ = cell_forward(
                cell=single_cell,
                inputs=hidden_output,
                index_layer=i)

            if use_layernorm:
                hidden_output = layer_norm(hidden_output, i)

        return hidden_output, len_feas
