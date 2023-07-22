def write_to_tb(writer, index, net, scalars={}, images={}):

    for scalar_name in scalars.keys():
        writer.add_scalar(scalar_name, scalars[scalar_name], index)

    for image_name in images.keys():
        writer.add_image(image_name, images[image_name], index)

    for name, parameter in net.named_parameters():
        if parameter.requires_grad and not isinstance(parameter.grad, type(None)):
            writer.add_histogram(name, parameter, index)
            writer.add_histogram(f"{name}.grad", parameter.grad, index)


def accuracy(predictions, targets):
    return (predictions == targets).sum()/targets.shape[0]