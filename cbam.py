    def channel_attention(self, input_feature, index, ratio=0.125):
        channel = int(input_feature.shape[1])
        size = int(channel * ratio)
        size = 1 if size < 1 else size
        # avg path
        avg_path = fluid.layers.pool2d(name='cbam_avg_pooling_' + str(index), input=input_feature, global_pooling=True, pool_type='avg', pool_padding='SAME')
        # learning_rate=0.1, regularizer=fluid.regularizer.L2Decay(1.0),
        avg_path = fluid.layers.fc(name='cbam_avg_fc1_' + str(index), input=avg_path, size=size, param_attr=ParamAttr(name='fc_share_1_' + str(index),  trainable=True), act='relu')
        avg_path = fluid.layers.fc(name='cbam_avg_fc2_' + str(index), input=avg_path, size=channel, param_attr=ParamAttr(name='fc_share_2_' + str(index), trainable=True), act='relu')
        avg_path = fluid.layers.reshape(x=avg_path, shape=[-1, channel, 1, 1])
        # max path
        max_path = fluid.layers.pool2d(name='cbam_max_pooling_' + str(index), input=input_feature, global_pooling=True, pool_type='max', pool_padding='SAME')
        max_path = fluid.layers.fc(name='cbam_max_fc1_' + str(index), input=max_path, size=size, param_attr=ParamAttr(name='fc_share_1_' + str(index), trainable=True), act='relu')
        max_path = fluid.layers.fc(name='cbam_max_fc2_' + str(index), input=max_path, size=channel, param_attr=ParamAttr(name='fc_share_2_' + str(index), trainable=True), act='relu')
        max_path = fluid.layers.reshape(x=max_path, shape=[-1, channel, 1, 1])
        # add
        output = fluid.layers.elementwise_add(avg_path, max_path, act='sigmoid')
        return fluid.layers.elementwise_mul(output, input_feature)


    def spatial_attention(self, channel_feature, index):
        max_feature = fluid.layers.reduce_max(input=channel_feature, dim=1, keep_dim=True)
        avg_feature = fluid.layers.reduce_mean(input=channel_feature, dim=1, keep_dim=True)
        concat_feature = fluid.layers.concat(input=[max_feature, avg_feature], axis=1)
        output = fluid.layers.conv2d(name='cbam_conv2d_' + str(index), input=concat_feature, num_filters=1, filter_size=(3, 3), padding="SAME",
                                     act="sigmoid")
        return fluid.layers.elementwise_mul(output, channel_feature)


    def cbam_module(self, input, index):
        channel_refined_feature = self.channel_attention(input, index)
        cbam_feature = self.spatial_attention(channel_refined_feature, index)
        return cbam_feature
